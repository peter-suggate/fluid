import assert from "node:assert/strict";
import test from "node:test";
import {
  brickBounds,
  applyRigidLowRankPressureOperator,
  buildHierarchy,
  buildSceneHierarchy,
  conservativeChildFractions,
  leafFaceNeighbors,
  restrictFaceFluxes,
  restrictVolumeFractions
} from "../lib/hierarchical-grid";
import { cloneScene, defaultScene } from "../lib/model";
import { createGPUHierarchyLayout, GPU_BRICK_META_WORDS, GPU_CELLS_PER_BRICK, GPU_CELL_FLOATS, rebuildGPUHierarchyFromTags, rebuildGPUHierarchyLayout } from "../lib/webgpu-hierarchy-layout";
import { initializeRigidBodies } from "../lib/rigid-body";

test("H11-01 one hierarchy level is exactly a uniform leaf-brick covering", () => {
  const scene = cloneScene(defaultScene);
  scene.hierarchy.levels = 1;
  scene.hierarchy.brickSize = 4;
  scene.hierarchy.maxActiveBricks = 100_000;
  const topology = buildHierarchy(scene);
  const expected = topology.finestBrickDims.x * topology.finestBrickDims.y * topology.finestBrickDims.z;
  assert.equal(topology.leaves.length, expected);
  assert.ok(topology.leaves.every((brick) => brick.level === 0 && brick.childIds.length === 0));
  assert.deepEqual(topology.baseBrickDims, topology.finestBrickDims);
  assert.equal(new Set(topology.pageTable.brickIds).size, expected);
});

test("H11-02 adaptive topology is complete, deterministic, and 2:1 balanced", () => {
  const scene = cloneScene(defaultScene);
  scene.hierarchy.levels = 4;
  scene.hierarchy.brickSize = 2;
  scene.hierarchy.maxActiveBricks = 100_000;
  scene.nominalResolution.length_m = 0.075;
  const oracle = (_brick: { level: number }, bounds: ReturnType<typeof brickBounds>) => bounds.min.x < -0.15 && bounds.min.y < 0.45;
  const a = buildHierarchy(scene, oracle);
  const b = buildHierarchy(scene, oracle);
  assert.deepEqual(a.leaves, b.leaves);
  assert.deepEqual(a.pageTable.brickIds, b.pageTable.brickIds);
  assert.equal(a.pageTable.brickIds.length, a.finestBrickDims.x * a.finestBrickDims.y * a.finestBrickDims.z);
  for (const leaf of a.leaves) for (const axis of [0, 1, 2] as const) for (const sign of [-1, 1] as const) {
    for (const id of leafFaceNeighbors(a, leaf, axis, sign)) assert.ok(Math.abs(leaf.level - a.bricks[id].level) <= 1);
  }
});

test("H11-03 refinement and restriction preserve VOF volume to roundoff", () => {
  for (const fraction of [0, 0.001, 0.15, 0.5, 0.91, 0.999, 1]) {
    const children = conservativeChildFractions(fraction, [-2, -1, 0, 1, 2, 1, 0, -1]);
    assert.ok(children.every((value) => value >= 0 && value <= 1));
    assert.ok(Math.abs(restrictVolumeFractions(children) - fraction) < 1e-12);
  }
  assert.equal(restrictFaceFluxes([0.1, -0.2, 0.3, 0.4]), 0.6);
});

test("H11-04 scene oracle refines initial surfaces and rigid-body neighborhoods", () => {
  const scene = cloneScene(defaultScene);
  scene.hierarchy.levels = 3;
  scene.hierarchy.brickSize = 2;
  scene.hierarchy.maxActiveBricks = 100_000;
  scene.nominalResolution.length_m = 0.05;
  const topology = buildSceneHierarchy(scene);
  assert.ok(topology.leaves.some((brick) => brick.level === scene.hierarchy.levels - 1));
  assert.ok(topology.leaves.some((brick) => brick.level < scene.hierarchy.levels - 1));
  assert.equal(topology.saturated, false);
});

test("H11-05 brick budget saturation is explicit and never over-allocates", () => {
  const scene = cloneScene(defaultScene);
  scene.hierarchy.levels = 4;
  scene.hierarchy.brickSize = 2;
  scene.nominalResolution.length_m = 0.075;
  const base = buildHierarchy({ ...scene, hierarchy: { ...scene.hierarchy, levels: 1 } });
  scene.hierarchy.maxActiveBricks = base.leaves.length + 7;
  const topology = buildHierarchy(scene, () => true);
  assert.equal(topology.saturated, true);
  assert.ok(topology.leaves.length <= scene.hierarchy.maxActiveBricks);
});

test("H11-06 GPU layout uses the same allocation path for uniform and adaptive grids", () => {
  const uniformScene = cloneScene(defaultScene);
  uniformScene.hierarchy.levels = 1;
  const uniform = createGPUHierarchyLayout(uniformScene, "balanced");
  assert.equal(uniform.activeCellCount, uniform.equivalentUniformCells);
  assert.equal(uniform.initialCells.length, uniform.topology.leaves.length * GPU_CELLS_PER_BRICK * 8);
  assert.ok([...uniform.pageTable].every((slot) => slot < uniform.topology.leaves.length));

  const adaptiveScene = cloneScene(defaultScene);
  adaptiveScene.hierarchy.levels = 3;
  const adaptive = createGPUHierarchyLayout(adaptiveScene, "balanced");
  assert.ok(adaptive.activeCellCount < adaptive.equivalentUniformCells);
  assert.ok(adaptive.topology.leaves.some((brick) => brick.level < 2));
  assert.ok(adaptive.topology.leaves.some((brick) => brick.level === 2));
});

test("H11-07 dynamic regridding preserves represented volume and finite state", () => {
  const scene=cloneScene(defaultScene);scene.hierarchy.levels=3;scene.hierarchy.interfaceHaloCells=1;scene.rigidBodies=[];
  const before=createGPUHierarchyLayout(scene,"balanced"),cells=new Float32Array(before.initialCells);
  // Introduce a resolved interface away from the initial dam so the oracle has
  // to move refinement rather than reproduce only the initialization layout.
  for(let index=0;index<before.activeCellCount;index+=1)cells[index*GPU_CELL_FLOATS+3]=index%11===0?0.35:index%7===0?1:0;
  const volume=(layout:typeof before,data:Float32Array)=>{let total=0;for(let slot=0;slot<layout.topology.leaves.length;slot+=1){const scale=layout.leafMetadata[slot*GPU_BRICK_META_WORDS+3],cellVolume=(scale*layout.topology.finestCellLength_m)**3;for(let local=0;local<GPU_CELLS_PER_BRICK;local+=1)total+=data[(slot*GPU_CELLS_PER_BRICK+local)*GPU_CELL_FLOATS+3]*cellVolume;}return total;};
  const expected=volume(before,cells),after=rebuildGPUHierarchyLayout(scene,"balanced",before,cells,initializeRigidBodies([])),actual=volume(after,after.initialCells);
  assert.ok(Math.abs(actual-expected)<=Math.max(1e-9,expected*2e-7),`${actual} != ${expected}`);
  assert.ok(after.initialCells.every(Number.isFinite));
  assert.ok(after.topology.leaves.some((brick)=>brick.level===scene.hierarchy.levels-1));
});

test("H11-08 coarse-fine leaf adjacency is reciprocal", () => {
  const scene=cloneScene(defaultScene);scene.hierarchy.levels=4;scene.hierarchy.brickSize=2;scene.nominalResolution.length_m=.075;scene.hierarchy.maxActiveBricks=100_000;
  const topology=buildHierarchy(scene,(_brick,bounds)=>bounds.min.x<-.1&&bounds.min.y<.5);
  for(const leaf of topology.leaves)for(const axis of [0,1,2] as const)for(const sign of [-1,1] as const)for(const neighborId of leafFaceNeighbors(topology,leaf,axis,sign)){
    assert.ok(leafFaceNeighbors(topology,topology.bricks[neighborId],axis,sign===1?-1:1).includes(leaf.id),`missing reciprocal face ${leaf.id} -> ${neighborId}`);
  }
});

test("H11-09 pressure-level rigid Schur term is symmetric positive semidefinite", () => {
  const jacobian=[
    [1,.2,0,0,.3,-.1],[-.4,1,.1,.2,0,.5],[.3,-.2,1,-.4,.2,0],[0,.6,-.3,1,-.2,.4]
  ];
  const inverseMassInertia=[.5,.5,.5,.2,.3,.4],x=[.7,-1.1,.2,.9],y=[-.3,.4,1.2,-.8];
  const ax=applyRigidLowRankPressureOperator(jacobian,inverseMassInertia,x),ay=applyRigidLowRankPressureOperator(jacobian,inverseMassInertia,y);
  const dot=(a:readonly number[],b:readonly number[])=>a.reduce((sum,value,index)=>sum+value*b[index],0);
  assert.ok(Math.abs(dot(x,ay)-dot(y,ax))<1e-12);
  assert.ok(dot(x,ax)>=-1e-12);
  assert.ok(dot(y,ay)>=-1e-12);
});

test("H11-10 GPU page tags produce deterministic balanced CPU topology",()=>{
  const scene=cloneScene(defaultScene);scene.hierarchy.levels=3;scene.rigidBodies=[];const before=createGPUHierarchyLayout(scene,"balanced"),tags=new Uint32Array(before.pageTable.length),dims=before.topology.finestBrickDims,centre=Math.floor(dims.x/2)+dims.x*(Math.floor(dims.y/2)+dims.y*Math.floor(dims.z/2));tags[centre]=scene.hierarchy.levels-1;
  const a=rebuildGPUHierarchyFromTags(scene,"balanced",before,tags,[],true),b=rebuildGPUHierarchyFromTags(scene,"balanced",before,tags,[],true);
  assert.deepEqual(a.leafMetadata,b.leafMetadata);assert.equal(a.topology.pageTable.levels[centre],scene.hierarchy.levels-1);
  for(const leaf of a.topology.leaves)for(const axis of [0,1,2] as const)for(const sign of [-1,1] as const)for(const id of leafFaceNeighbors(a.topology,leaf,axis,sign))assert.ok(Math.abs(leaf.level-a.topology.bricks[id].level)<=1);
});
