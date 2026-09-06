import assert from "node:assert/strict";
import test from "node:test";
import { planAdaptiveSparseBrickOctree, planAdaptiveSparseBrickOctreeSteps } from "../lib/core/adaptive-sparse-brick-plan";
import { mortonEncode3D, SPARSE_BRICK_VOXEL_TERMINAL } from "../lib/svo/sparse-brick-octree";
import { driveCooperativeBuild } from "../lib/core/cooperative-build";

test("occupancy-backed plans preserve explicit proxy union and solver ownership", async () => {
  const selected = Array.from({length:70},(_,i)=>({x:(i*7)%16,y:(i*11)%16,z:Math.floor(i/9)}));
  const explicit = [{x:15,y:15,z:15},{x:0,y:1,z:2}];
  const occupancy = {
    keys(level:number){const scale=2**(4-level);return new Set(selected.map(p=>
      mortonEncode3D(Math.floor(p.x/scale),Math.floor(p.y/scale),Math.floor(p.z/scale))));},
    has(level:number,p:{x:number;y:number;z:number}){const scale=2**(4-level);return selected.some(q=>
      Math.floor(q.x/scale)===p.x&&Math.floor(q.y/scale)===p.y&&Math.floor(q.z/scale)===p.z);},
  };
  for(const solverBricks of [[],[{x:1,y:1,z:1}]]){
    const options={brickSize:8 as const,maximumDepth:4,solverLevel:2,maximumEnvironmentCoarseningPower:1,solverBricks,
      refineEnvironmentLeaf:(level:number,p:{x:number;y:number;z:number})=>level<2||p.x%2===0};
    assert.deepEqual(planAdaptiveSparseBrickOctree({...options,proxyBricks:explicit,proxyOccupancy:occupancy}),
      planAdaptiveSparseBrickOctree({...options,proxyBricks:[...selected,...explicit]}));
    assert.deepEqual(await driveCooperativeBuild(planAdaptiveSparseBrickOctreeSteps({...options,proxyBricks:explicit,proxyOccupancy:occupancy,
      refineEnvironmentLeaf:undefined,classifyEnvironmentBatch:async(level,coordinates)=>coordinates.map(p=>({
        refine:options.refineEnvironmentLeaf(level,p),terminal:SPARSE_BRICK_VOXEL_TERMINAL}))})),
      planAdaptiveSparseBrickOctree({...options,proxyBricks:[...selected,...explicit]}));
  }
});
