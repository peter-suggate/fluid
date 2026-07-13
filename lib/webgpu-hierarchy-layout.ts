import { damBreakFractions } from "./initial-fluid";
import { brickCellLength, buildHierarchy, buildSceneHierarchy, type HierarchyTopology, type Int3 } from "./hierarchical-grid";
import { cloneScene, type SceneDescription } from "./model";
import type { RigidBodyState } from "./rigid-body";
import type { GPUQuality } from "./webgpu-eulerian";

export const GPU_BRICK_SIZE = 4;
export const GPU_CELLS_PER_BRICK = GPU_BRICK_SIZE ** 3;
export const GPU_CELL_FLOATS = 8;
export const GPU_BRICK_META_WORDS = 8;

const qualityScale: Record<GPUQuality, number> = { balanced: 2, high: 0.72, ultra: 0.55 };

export interface GPUHierarchyLayout {
  topology: HierarchyTopology;
  leafMetadata: Uint32Array;
  pageTable: Uint32Array;
  initialCells: Float32Array;
  nodeToLeafSlot: Int32Array;
  equivalentUniformCells: number;
  activeCellCount: number;
  physicalFinestCellDims: Int3;
}

function occupancy(scene: SceneDescription, x: number, y: number, z: number): number {
  const c = scene.container;
  if (x < -c.width_m / 2 || x >= c.width_m / 2 || y < 0 || y >= c.height_m || z < -c.depth_m / 2 || z >= c.depth_m / 2) return 0;
  if (scene.fluid.initialCondition === "tank-fill") return y <= c.height_m * c.fillFraction ? 1 : 0;
  const dam = damBreakFractions(c.fillFraction);
  return x <= -c.width_m / 2 + c.width_m * dam.width && y <= c.height_m * dam.height && z <= -c.depth_m / 2 + c.depth_m * dam.depth ? 1 : 0;
}

export function effectiveHierarchyScene(scene: SceneDescription, quality: GPUQuality): SceneDescription {
  const effective = cloneScene(scene),scale=qualityScale[quality];
  effective.hierarchy.brickSize = GPU_BRICK_SIZE;
  effective.nominalResolution.length_m *= scale;
  effective.hierarchy.interfaceHaloCells=Math.max(GPU_BRICK_SIZE,Math.ceil(scene.hierarchy.interfaceHaloCells/Math.sqrt(scale)));
  effective.hierarchy.solidHaloCells=Math.max(1,Math.ceil(scene.hierarchy.solidHaloCells/scale));
  if(quality==="balanced")effective.hierarchy.regridInterval*=8;
  return effective;
}

function encodeLayout(scene: SceneDescription, topology: HierarchyTopology, cells?: Float32Array): GPUHierarchyLayout {
  const nodeToLeafSlot = new Int32Array(topology.bricks.length).fill(-1);
  topology.leaves.forEach((brick, slot) => { nodeToLeafSlot[brick.id] = slot; });
  const pageTable = new Uint32Array(topology.pageTable.brickIds.length);
  for (let index = 0; index < pageTable.length; index += 1) {
    const slot = nodeToLeafSlot[topology.pageTable.brickIds[index]];
    if (slot < 0) throw new Error("Hierarchy page table references a non-leaf node");
    pageTable[index] = slot;
  }
  const leafMetadata = new Uint32Array(topology.leaves.length * GPU_BRICK_META_WORDS);
  const initialCells = cells ?? new Float32Array(topology.leaves.length * GPU_CELLS_PER_BRICK * GPU_CELL_FLOATS);
  let initialVolume = 0;
  topology.leaves.forEach((brick, slot) => {
    const scale = 2 ** (topology.settings.levels - 1 - brick.level);
    const origin = {
      x: brick.coord.x * GPU_BRICK_SIZE * scale,
      y: brick.coord.y * GPU_BRICK_SIZE * scale,
      z: brick.coord.z * GPU_BRICK_SIZE * scale
    };
    const meta = slot * GPU_BRICK_META_WORDS;
    leafMetadata.set([origin.x, origin.y, origin.z, scale, brick.level, brick.id, brick.parentId, 1], meta);
    const h = brickCellLength(topology, brick.level);
    for (let z = 0; z < GPU_BRICK_SIZE; z += 1) for (let y = 0; y < GPU_BRICK_SIZE; y += 1) for (let x = 0; x < GPU_BRICK_SIZE; x += 1) {
      const world = {
        x: topology.origin_m.x + (origin.x + (x + 0.5) * scale) * topology.finestCellLength_m,
        y: topology.origin_m.y + (origin.y + (y + 0.5) * scale) * topology.finestCellLength_m,
        z: topology.origin_m.z + (origin.z + (z + 0.5) * scale) * topology.finestCellLength_m
      };
      const cell = (slot * GPU_CELLS_PER_BRICK + x + GPU_BRICK_SIZE * (y + GPU_BRICK_SIZE * z)) * GPU_CELL_FLOATS;
      const alpha = cells ? initialCells[cell + 3] : occupancy(scene, world.x, world.y, world.z);
      // negative-face velocity xyz, alpha; positive-face velocity xyz, pressure
      if (!cells) initialCells[cell + 3] = alpha;
      initialVolume += alpha * h ** 3;
    }
  });
  // The first reduction reference is stored in cubic metres by the solver; keep
  // it discoverable without adding another mutable layout field.
  void initialVolume;
  return {
    topology,
    leafMetadata,
    pageTable,
    initialCells,
    nodeToLeafSlot,
    equivalentUniformCells: topology.paddedFinestCellDims.x * topology.paddedFinestCellDims.y * topology.paddedFinestCellDims.z,
    activeCellCount: topology.leaves.length * GPU_CELLS_PER_BRICK,
    physicalFinestCellDims: {
      x: Math.ceil(scene.container.width_m / topology.finestCellLength_m),
      y: Math.ceil(scene.container.height_m / topology.finestCellLength_m),
      z: Math.ceil(scene.container.depth_m / topology.finestCellLength_m)
    }
  };
}

export function createGPUHierarchyLayout(scene: SceneDescription, quality: GPUQuality): GPUHierarchyLayout {
  const effective = effectiveHierarchyScene(scene, quality);
  return encodeLayout(scene, buildSceneHierarchy(effective));
}

function locateLayoutCellIndex(layout: GPUHierarchyLayout, x: number, y: number, z: number): number {
  const d = layout.topology.paddedFinestCellDims;
  if (x < 0 || y < 0 || z < 0 || x >= d.x || y >= d.y || z >= d.z) return -1;
  const pageX = Math.floor(x / GPU_BRICK_SIZE), pageY = Math.floor(y / GPU_BRICK_SIZE), pageZ = Math.floor(z / GPU_BRICK_SIZE);
  const pd = layout.topology.finestBrickDims;
  const slot = layout.pageTable[pageX + pd.x * (pageY + pd.y * pageZ)];
  const meta = slot * GPU_BRICK_META_WORDS, scale = layout.leafMetadata[meta + 3];
  const localX = Math.min(GPU_BRICK_SIZE - 1, Math.floor((x - layout.leafMetadata[meta]) / scale));
  const localY = Math.min(GPU_BRICK_SIZE - 1, Math.floor((y - layout.leafMetadata[meta + 1]) / scale));
  const localZ = Math.min(GPU_BRICK_SIZE - 1, Math.floor((z - layout.leafMetadata[meta + 2]) / scale));
  return slot * GPU_CELLS_PER_BRICK + localX + GPU_BRICK_SIZE * (localY + GPU_BRICK_SIZE * localZ);
}

function mark(tags: Uint8Array, dims: Int3, pointX: number, pointY: number, pointZ: number, level: number, halo: number): void {
  for (let z = Math.max(0, pointZ - halo); z <= Math.min(dims.z - 1, pointZ + halo); z += 1) for (let y = Math.max(0, pointY - halo); y <= Math.min(dims.y - 1, pointY + halo); y += 1) for (let x = Math.max(0, pointX - halo); x <= Math.min(dims.x - 1, pointX + halo); x += 1) {
    const index = x + dims.x * (y + dims.y * z);tags[index] = Math.max(tags[index], level);
  }
}

/**
 * Rebuilds leaf topology from the current field. Transfers are evaluated on the
 * finest logical lattice, so restriction and prolongation conserve VOF volume
 * exactly for piecewise-constant leaf data.
 */
export function rebuildGPUHierarchyLayout(scene: SceneDescription, quality: GPUQuality, previous: GPUHierarchyLayout, currentCells: Float32Array, bodies: readonly RigidBodyState[], allowCoarsening = true): GPUHierarchyLayout {
  const effective = effectiveHierarchyScene(scene, quality), levels = effective.hierarchy.levels, pageDims = previous.topology.finestBrickDims;
  const tags = new Uint8Array(pageDims.x * pageDims.y * pageDims.z);
  if(!allowCoarsening)for(let index=0;index<tags.length;index+=1)tags[index]=previous.topology.pageTable.levels[index];
  const halo = Math.ceil(effective.hierarchy.interfaceHaloCells / GPU_BRICK_SIZE);
  const fineDims = previous.topology.paddedFinestCellDims;
  for (let z = 0; z < fineDims.z; z += 1) for (let y = 0; y < fineDims.y; y += 1) for (let x = 0; x < fineDims.x; x += 1) {
    const here = locateLayoutCellIndex(previous,x,y,z);if (here<0) continue;const offset = here * GPU_CELL_FLOATS, alpha = currentCells[offset + 3];
    if (alpha <= 1e-4 && effective.hierarchy.minimumFluidLevel === 0) continue;
    let interfaceCell = alpha > 1e-4 && alpha < 1 - 1e-4,velocityDetail=0;
    const velocityX=0.5*(currentCells[offset]+currentCells[offset+4]),velocityY=0.5*(currentCells[offset+1]+currentCells[offset+5]),velocityZ=0.5*(currentCells[offset+2]+currentCells[offset+6]);
    for(let neighbor=0;neighbor<6;neighbor+=1){
      const axis=Math.floor(neighbor/2),sign=neighbor%2===0?1:-1,other=locateLayoutCellIndex(previous,x+(axis===0?sign:0),y+(axis===1?sign:0),z+(axis===2?sign:0));const otherAlpha=other>=0?currentCells[other*GPU_CELL_FLOATS+3]:0;if(Math.abs(alpha-otherAlpha)>0.05)interfaceCell=true;if(other>=0){const o=other*GPU_CELL_FLOATS,otherX=0.5*(currentCells[o]+currentCells[o+4]),otherY=0.5*(currentCells[o+1]+currentCells[o+5]),otherZ=0.5*(currentCells[o+2]+currentCells[o+6]);velocityDetail=Math.max(velocityDetail,Math.hypot(velocityX-otherX,velocityY-otherY,velocityZ-otherZ));}
    }
    const characteristic=Math.max(0.1,Math.hypot(velocityX,velocityY,velocityZ)),dynamicCell=velocityDetail/characteristic>effective.hierarchy.velocityErrorTolerance;
    const desired = interfaceCell||dynamicCell ? levels - 1 : alpha>1e-4?Math.max(effective.hierarchy.minimumFluidLevel,0):0;
    const travelPages=Math.min(2,Math.ceil(characteristic*scene.numerics.fixedDt_s*effective.hierarchy.regridInterval/(previous.topology.finestCellLength_m*GPU_BRICK_SIZE)));
    if (desired > 0) mark(tags,pageDims,Math.floor(x/GPU_BRICK_SIZE),Math.floor(y/GPU_BRICK_SIZE),Math.floor(z/GPU_BRICK_SIZE),desired,interfaceCell?halo+travelPages:0);
  }
  const h = previous.topology.finestCellLength_m;
  for (const body of bodies) {
    const d=body.description.dimensions_m,r=body.description.shape==="sphere"?d.x:body.description.shape==="box"?Math.hypot(d.x,d.y,d.z)/2:Math.hypot(d.x,d.y/2),padding=effective.hierarchy.solidHaloCells*h;
    const min={x:body.position_m.x-r-padding,y:body.position_m.y-r-padding,z:body.position_m.z-r-padding},max={x:body.position_m.x+r+padding,y:body.position_m.y+r+padding,z:body.position_m.z+r+padding};
    const origin=previous.topology.origin_m;
    const lo={x:Math.floor((min.x-origin.x)/(h*GPU_BRICK_SIZE)),y:Math.floor((min.y-origin.y)/(h*GPU_BRICK_SIZE)),z:Math.floor((min.z-origin.z)/(h*GPU_BRICK_SIZE))};
    const hi={x:Math.floor((max.x-origin.x)/(h*GPU_BRICK_SIZE)),y:Math.floor((max.y-origin.y)/(h*GPU_BRICK_SIZE)),z:Math.floor((max.z-origin.z)/(h*GPU_BRICK_SIZE))};
    for(let z=lo.z;z<=hi.z;z+=1)for(let y=lo.y;y<=hi.y;y+=1)for(let x=lo.x;x<=hi.x;x+=1)mark(tags,pageDims,x,y,z,levels-1,0);
  }
  const topology=buildHierarchy(effective,(brick)=>{const scale=2**(levels-1-brick.level),start={x:brick.coord.x*scale,y:brick.coord.y*scale,z:brick.coord.z*scale};for(let z=start.z;z<start.z+scale;z+=1)for(let y=start.y;y<start.y+scale;y+=1)for(let x=start.x;x<start.x+scale;x+=1){if(tags[x+pageDims.x*(y+pageDims.y*z)]>brick.level)return true;}return false;});
  const blank=encodeLayout(scene,topology),accum=new Float64Array(blank.activeCellCount*GPU_CELL_FLOATS),counts=new Uint32Array(blank.activeCellCount);
  for(let z=0;z<fineDims.z;z+=1)for(let y=0;y<fineDims.y;y+=1)for(let x=0;x<fineDims.x;x+=1){const oldCell=locateLayoutCellIndex(previous,x,y,z),newCell=locateLayoutCellIndex(blank,x,y,z);if(oldCell<0||newCell<0)continue;const oldOffset=oldCell*GPU_CELL_FLOATS,newOffset=newCell*GPU_CELL_FLOATS;for(let component=0;component<GPU_CELL_FLOATS;component+=1)accum[newOffset+component]+=currentCells[oldOffset+component];counts[newCell]+=1;}
  const transferred=new Float32Array(blank.initialCells.length);for(let index=0;index<blank.activeCellCount;index+=1){const count=Math.max(1,counts[index]);for(let component=0;component<GPU_CELL_FLOATS;component+=1)transferred[index*GPU_CELL_FLOATS+component]=accum[index*GPU_CELL_FLOATS+component]/count;}
  return encodeLayout(scene,topology,transferred);
}

/** Builds compact CPU topology from page-sized refinement tags produced on GPU. */
export function rebuildGPUHierarchyFromTags(scene: SceneDescription, quality: GPUQuality, previous: GPUHierarchyLayout, desiredTags: Uint32Array, bodies: readonly RigidBodyState[], allowCoarsening = true): GPUHierarchyLayout {
  const effective=effectiveHierarchyScene(scene,quality),levels=effective.hierarchy.levels,pageDims=previous.topology.finestBrickDims,pageCount=pageDims.x*pageDims.y*pageDims.z;
  if(desiredTags.length!==pageCount)throw new Error(`GPU hierarchy tag count ${desiredTags.length} does not match ${pageCount} pages`);
  const tags=new Uint8Array(pageCount);for(let index=0;index<pageCount;index+=1)tags[index]=Math.max(Math.min(levels-1,desiredTags[index]),allowCoarsening?0:previous.topology.pageTable.levels[index]);
  const h=previous.topology.finestCellLength_m;
  for(const body of bodies){
    const d=body.description.dimensions_m,r=body.description.shape==="sphere"?d.x:body.description.shape==="box"?Math.hypot(d.x,d.y,d.z)/2:Math.hypot(d.x,d.y/2),padding=effective.hierarchy.solidHaloCells*h;
    const origin=previous.topology.origin_m,lo={x:Math.floor((body.position_m.x-r-padding-origin.x)/(h*GPU_BRICK_SIZE)),y:Math.floor((body.position_m.y-r-padding-origin.y)/(h*GPU_BRICK_SIZE)),z:Math.floor((body.position_m.z-r-padding-origin.z)/(h*GPU_BRICK_SIZE))},hi={x:Math.floor((body.position_m.x+r+padding-origin.x)/(h*GPU_BRICK_SIZE)),y:Math.floor((body.position_m.y+r+padding-origin.y)/(h*GPU_BRICK_SIZE)),z:Math.floor((body.position_m.z+r+padding-origin.z)/(h*GPU_BRICK_SIZE))};
    for(let z=lo.z;z<=hi.z;z+=1)for(let y=lo.y;y<=hi.y;y+=1)for(let x=lo.x;x<=hi.x;x+=1)mark(tags,pageDims,x,y,z,levels-1,0);
  }
  const topology=buildHierarchy(effective,(brick)=>{const scale=2**(levels-1-brick.level),startX=brick.coord.x*scale,startY=brick.coord.y*scale,startZ=brick.coord.z*scale;for(let z=startZ;z<startZ+scale;z+=1)for(let y=startY;y<startY+scale;y+=1)for(let x=startX;x<startX+scale;x+=1)if(tags[x+pageDims.x*(y+pageDims.y*z)]>brick.level)return true;return false;});
  return encodeLayout(scene,topology);
}
