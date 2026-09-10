import assert from "node:assert/strict";
import test from "node:test";
import { createSvoRenderTerrainRefinement } from "../lib/svo/features/scene-publication/svo-render-solid-field";

test("terrain refinement includes interpolated height influence across brick boundaries",()=>{
 const heights=new Float32Array(8);heights[3]=5;
 const refinement=createSvoRenderTerrainRefinement({
  field:{origin_m:[0,0],cellSize_m:[1,1],dimensions:[8,1],heights_m:heights,materialId:1,patches:[]},
  worldOrigin_m:[0,0,0],renderCellSize_m:[1,1,1],refinedBrickDimensions:[4,4,1],
  nodeEdge_m:[[4,4,4],[2,2,2]],brickSize:2,maximumDepth:1,
 });
 // The height at x=3.5 contributes to [4,4.5], across the node boundary.
 assert.equal(refinement.refineEnvironmentLeaf(0,{x:1,y:1,z:0}),true);
 assert.equal(refinement.refineEnvironmentLeaf(0,{x:1,y:2,z:0}),false,"nodes entirely above the influence remain coarse");
});
