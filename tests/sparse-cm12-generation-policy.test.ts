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

test("min8 survives requested refinement and grades the outside through coarsening", () => {
 const atlas = createSparseAdaptiveMassAtlas([16,8,8], [0,1].map(x => ({
  key:x, coordinate:[x,0,0] as const, resolution:8 as const,
  density:new Float64Array(512).fill(1), gamma:new Float64Array(512).fill(1),
 })),0,8);
 const plan=planSparseCM12ResidentGeneration(atlas,new Set([0,1]),new Map([
  [0,{resolution:8 as const,mergeable:false,minimumCellWidth:8,maximumCellWidth:1}],
  [1,{resolution:8 as const,mergeable:false,maximumCellWidth:1}],
 ]),limits);
 assert.equal(plan?.status,"ready");
 if(plan?.status!=="ready")return;
 assert.equal(plan.atlas.directory.get(0)!.resolution,1);
 assert.equal(plan.atlas.directory.get(1)!.resolution,2);
});

test("a global min8 floor applies to every macro descendant", () => {
 const atlas=createSparseAdaptiveMassAtlas([32,32,32],[{
  key:0,coordinate:[0,0,0],spanBricks:4,resolution:8,
  density:new Float64Array(512).fill(1),gamma:new Float64Array(512).fill(1),
 }],0,8);
 const plan=planSparseCM12ResidentGeneration(atlas,new Set([0]),new Map([[0,{
  resolution:8 as const,mergeable:false,minimumCellWidth:8,maximumCellWidth:1,
 }]]),limits);
 assert.equal(plan?.status,"ready");
 if(plan?.status!=="ready")return;
 assert.ok(plan.atlas.bricks.every(b=>8*sparseBrickSpan(b)/b.resolution>=8));
});

test("authored min32 groups non-quiet clipped surface coverage without filling holes", () => {
 const bricks=Array.from({length:48},(_,id)=>({
  key:sparseBrickKey([id%4,Math.floor(id/4)%4,Math.floor(id/16)],[4,4,3]),
  coordinate:[id%4,Math.floor(id/4)%4,Math.floor(id/16)] as const,
  resolution:8 as const,density:new Float64Array(512).fill(0.4),gamma:new Float64Array(512).fill(1),
 }));
 const atlas=createSparseAdaptiveMassAtlas([32,32,20],bricks,0,8);
 const active=new Set(bricks.map(b=>b.key));
 const intents=new Map(bricks.map(b=>[b.key,{resolution:8 as const,mergeable:false,minimumCellWidth:32}]));
 const plan=planSparseCM12ResidentGeneration(atlas,active,intents,limits);
 assert.equal(plan?.status,"ready");
 if(plan?.status!=="ready")return;
 assert.equal(plan.atlas.bricks.length,1);
 assert.equal(sparseBrickSpan(plan.atlas.bricks[0]!),4);
 assert.equal(plan.atlas.bricks[0]!.resolution,1);
 const incomplete=createSparseAdaptiveMassAtlas([32,32,20],bricks.slice(1),0,8);
 assert.equal(planSparseCM12ResidentGeneration(incomplete,active,intents,limits)?.status,"deferred");
});

test("requested dormant receiver refinement survives reclamation without activating unrelated air", () => {
 const atlas=createSparseAdaptiveMassAtlas([24,8,8],[0,1,2].map(x=>({
  key:x,coordinate:[x,0,0] as const,resolution:(x===0?8:2) as 8|2,
  density:new Float64Array(x===0?512:8).fill(x===0?1:0),
  gamma:new Float64Array(x===0?512:8).fill(1),
 })),0,8,false,false);
 const plan=planSparseCM12ResidentGeneration(atlas,new Set([0]),new Map([
  [1,{resolution:4 as const,mergeable:false}],
 ]),limits);
 assert.equal(plan?.status,"ready");
 if(plan?.status!=="ready")return;
 assert.equal(plan.atlas.directory.get(1)?.resolution,4);
 assert.equal(plan.atlas.directory.has(2),false);
 assert.deepEqual([...plan.active],[0]);
 assert.ok(plan.atlas.directory.get(1)!.density.every(value=>value===0));
 assert.equal(planSparseCM12ResidentGeneration(atlas,new Set([0]),new Map([
  [1,{resolution:4 as const,mergeable:false}],
 ]),{...limits,maximumCells:512})?.status,"deferred");
});
