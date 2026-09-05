import assert from "node:assert/strict";
import test from "node:test";
import { createSparseAdaptiveMassAtlas, sparseBrickKey, sparseBrickSpan } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { planSparseCM12ResidentGeneration } from "../lib/methods/adaptive-mass/sparse-cm12-generation-policy";
import { compileSparseCM12StableLeafFaceNeighbors } from "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-topology";
const limits = { maximumLeaves: 4096, maximumCells: 262144, maximumSpanBricks: 1024 };
test("quiet sibling coverage merges into physical 64h cells", () => {
 const atlas = createSparseAdaptiveMassAtlas([64,64,64], Array.from({length:8},(_,i) => {
  const coordinate = [4*(i&1),4*((i>>>1)&1),4*(i>>>2)] as const;
  return { key:sparseBrickKey(coordinate,[8,8,8]), coordinate, spanBricks:4, resolution:1 as const,
   density:new Float64Array([1]),gamma:new Float64Array([1]) };
 }),0,8);
 const active = new Set(atlas.bricks.map(b=>b.key));
 const intents = new Map(atlas.bricks.map(b=>[b.key,{resolution:1 as const,mergeable:true}]));
 const plan = planSparseCM12ResidentGeneration(atlas,active,intents,limits)!;
 assert.ok(plan.status === "ready");
 assert.equal(plan.atlas.bricks.length,1);
 assert.equal(8*sparseBrickSpan(plan.atlas.bricks[0]!)/plan.atlas.bricks[0]!.resolution,64);
 assert.deepEqual([...plan.active],[0]);
 intents.set(atlas.bricks[0]!.key,{resolution:1,mergeable:false});
 assert.equal(planSparseCM12ResidentGeneration(atlas,active,intents,limits),undefined);
});
test("local physical grading splits macro coverage without losing active descendants", () => {
 const atlas = createSparseAdaptiveMassAtlas([72,64,64],[{
  key:0,coordinate:[0,0,0],spanBricks:8,resolution:1,
  density:new Float64Array([1]),gamma:new Float64Array([1]),
 },{key:8,coordinate:[8,0,0],resolution:8,
  density:new Float64Array(512).fill(1),gamma:new Float64Array(512).fill(1),
 }],0,8);
 const plan = planSparseCM12ResidentGeneration(atlas,new Set([0,8]),new Map(),limits)!;
 assert.ok(plan.status === "ready");
 assert.ok(plan.atlas.bricks.length>2);
 assert.equal(plan.active.size,plan.atlas.bricks.length);
 const neighbors = compileSparseCM12StableLeafFaceNeighbors({coordinates:plan.atlas.bricks.map(b=>b.coordinate),spans:plan.atlas.bricks.map(sparseBrickSpan)});
 for (let id=0;id<plan.atlas.bricks.length;id++) for (const other of neighbors[id]!) {
  const a=plan.atlas.bricks[id]!,b=plan.atlas.bricks[other]!;
  const aw=8*sparseBrickSpan(a)/a.resolution,bw=8*sparseBrickSpan(b)/b.resolution;
  assert.ok(Math.max(aw,bw)<=2*Math.min(aw,bw));
 }
 assert.equal(planSparseCM12ResidentGeneration(atlas,new Set([0,8]),new Map(),{...limits,maximumLeaves:2})?.status,"deferred");
 assert.equal(atlas.bricks.length,2);
});
test("a macro's own physical demand splits beyond its finest local rung", () => {
 const atlas = createSparseAdaptiveMassAtlas([32,32,32],[{
  key:0,coordinate:[0,0,0],spanBricks:4,resolution:8,
  density:new Float64Array(512).fill(1),gamma:new Float64Array(512).fill(1),
 }],0,8);
 const intents = new Map([[0,{resolution:8 as const,mergeable:false,maximumCellWidth:1}]]);
 const plan = planSparseCM12ResidentGeneration(atlas,new Set([0]),intents,limits)!;
 assert.ok(plan.status === "ready");
 assert.equal(plan.atlas.bricks.length,64);
 assert.equal(plan.active.size,64);
 assert.ok(plan.atlas.bricks.every(b=>8*sparseBrickSpan(b)/b.resolution===1));
 assert.equal(plan.atlas.bricks.reduce((v,b)=>v+(8*sparseBrickSpan(b))**3,0),32**3);
 assert.equal(planSparseCM12ResidentGeneration(atlas,new Set([0]),intents,
  {...limits,maximumCells:1024})?.status,"deferred");
});
