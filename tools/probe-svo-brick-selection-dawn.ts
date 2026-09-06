/** Pair the production GPU selector with the independent CPU SDF oracle. */
import assert from "node:assert/strict";
import {createDawnRenderDevice} from "./svo-dry-frame-harness";
import {selectSvoBrickOccupancyGpu} from "../lib/svo/webgpu-svo-brick-selection";
import {mortonDecode3D} from "../lib/svo/sparse-brick-octree";
import {planAdaptiveSparseBrickOctree} from "../lib/core/adaptive-sparse-brick-plan";
import {buildSvoScenePrimitives,svoScenePrimitiveSolidReach} from "../lib/svo/svo-scene-primitives";
import {liveSceneBrickCoordinatesForRegions,liveSceneReachableBrickCoordinates} from "../lib/svo/webgpu-svo-sparse-bricks";
import {getScenePreset} from "../lib/core/scenes";
import {classifySvoNodesGpu} from "../lib/svo/webgpu-svo-node-classification";
import {buildSvoPlanarBoundaryCatalog,createSvoPlanarLeafClassifier} from "../lib/svo/svo-planar-boundary";
import {buildEnvironmentProxyCatalog,environmentProxyPrimitives} from "../lib/core/voxel-environments";

const {device,validationErrors}=await createDawnRenderDevice();
try {
  const build=buildSvoScenePrimitives(getScenePreset("garden-svo-lighting").create());
  const kinds=new Set<string>();
  let comparisons=0,extraBricks=0;
  for(const entry of build.metadata){
    const descriptor=build.descriptors[entry.primitiveIndex];
    if(kinds.has(descriptor.kind))continue;kinds.add(descriptor.kind);
    const isolated={...build,metadata:[entry]};
    const {min,max}=entry.coverageBounds.conservative_m;
    const cell=Math.max(max.x-min.x,max.y-min.y,max.z-min.z)/48;
    const worldOrigin=[min.x-cell*8,min.y-cell*8,min.z-cell*8] as const;
    const input={regions:[{minimum:[min.x,min.y,min.z] as const,maximum:[max.x,max.y,max.z] as const}],
      worldOrigin,cellSize:[cell,cell,cell] as const,brickSize:4,brickDimensions:[16,16,16] as const,maximumDepth:4};
    const cpu=liveSceneReachableBrickCoordinates(liveSceneBrickCoordinatesForRegions(input.regions,worldOrigin,input.cellSize,4,input.brickDimensions),
      svoScenePrimitiveSolidReach(isolated),worldOrigin,input.cellSize,4);
    const occupancy=await selectSvoBrickOccupancyGpu(device,isolated,input);
    const gpu=[...occupancy.keys(4)].map(mortonDecode3D);
    const planOptions={brickSize:4 as const,maximumDepth:4,maximumEnvironmentCoarseningPower:3,solverBricks:[],
      refineEnvironmentLeaf:(level:number,p:{x:number;y:number;z:number})=>level<3||p.x%2===0};
    assert.deepEqual(planAdaptiveSparseBrickOctree({...planOptions,proxyBricks:[],proxyOccupancy:occupancy}),
      planAdaptiveSparseBrickOctree({...planOptions,proxyBricks:gpu}),"GPU occupancy pyramid must preserve the complete adaptive plan");
    const key=(p:{x:number;y:number;z:number})=>`${p.x},${p.y},${p.z}`;
    const actual=new Set(gpu.map(key)),expected=new Set(cpu.map(key));
    const missing=cpu.filter(p=>!actual.has(key(p)));
    assert.deepEqual(missing,[],`${descriptor.kind}: GPU selection must retain every CPU-reachable brick`);
    const extras=gpu.filter(p=>!expected.has(key(p))).length;
    assert.ok(extras<=Math.max(2,cpu.length*0.01),`${descriptor.kind}: excess selection beyond f32 boundary tolerance (${extras})`);
    comparisons++;extraBricks+=extras;
    console.log(JSON.stringify({kind:descriptor.kind,cpu:cpu.length,gpu:gpu.length,extras}));
  }
  assert.ok(comparisons>=4,"exercise multiple primitive families");
  const classificationScene=getScenePreset("garden-svo-lighting").create();
  const primitives=environmentProxyPrimitives(buildEnvironmentProxyCatalog(classificationScene,classificationScene.environment??"default"),true);
  const catalog=buildSvoPlanarBoundaryCatalog(primitives,p=>({materialId:p.ownerIndex+32,ownerId:p.ownerIndex+12}));
  const planar={sources:catalog.sources,worldOrigin_m:[-2,-1,-2] as const,nodeEdge_m:[[4,4,4],[2,2,2],[1,1,1],[0.5,0.5,0.5]],
    blockers:primitives.map(p=>({minimum:[p.aabb_m.min.x,p.aabb_m.min.y,p.aabb_m.min.z] as const,
      maximum:[p.aabb_m.max.x,p.aabb_m.max.y,p.aabb_m.max.z] as const,planarSourceIndex:catalog.patchIndexByOwner.get(p.ownerIndex)}))};
  const classifier=createSvoPlanarLeafClassifier(planar);
  const coordinates=Array.from({length:512},(_,i)=>({x:i%8,y:Math.floor(i/8)%8,z:Math.floor(i/64)}));
  for(let level=0;level<4;level++){
    const actual=await classifySvoNodesGpu(device,{planar,coordinates,level,candidateCount:primitives.length,candidateLimit:64});
    const expected=coordinates.map(p=>{
      const edge=planar.nodeEdge_m[level],c=[p.x,p.y,p.z],lo=c.map((v,a)=>planar.worldOrigin_m[a]+v*edge[a]);
      const candidates=planar.blockers.filter(b=>b.minimum.every((v,a)=>v<=lo[a]+edge[a])&&b.maximum.every((v,a)=>v>=lo[a])).length;
      return {refine:classifier.requiresFineVoxelResidual(level,p)||candidates>64,terminal:classifier(level,p)};
    });
    assert.deepEqual(actual,expected,`GPU integer classification must match CPU inclusive bounds at level ${level}`);
  }
  assert.deepEqual(validationErrors,[]);
  console.log(JSON.stringify({comparisons,extraBricks,validationErrors}));
} finally {device.destroy();}
