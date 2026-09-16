import assert from "node:assert/strict";
import test from "node:test";
import { createSparseAdaptiveMassAtlas, sparseAtlasBrickKey, sparseBrickKey, sparseBrickSpan } from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { planSparseCM12ResidentGeneration } from "../lib/methods/adaptive-volume/sparse-cm12-generation-policy";
import { compileSparseCM12StableLeafFaceNeighbors } from "../lib/methods/adaptive-volume/sparse-cm12-factored-aei-topology";
const limits = { maximumLeaves: 4096, maximumCells: 262144, maximumSpanBricks: 1024 };
test("a new dry receiver expands to the face-patch span of a frozen macro host", () => {
 const dimensions=[32,32,32] as const;
 const coordinate=[4,1,1] as const;
 const key=sparseAtlasBrickKey(coordinate,{brickDimensions:[4,4,4],signedCoordinates:true});
 const hostKey=sparseAtlasBrickKey([0,0,0],{brickDimensions:[4,4,4],signedCoordinates:true});
 const atlas=createSparseAdaptiveMassAtlas(dimensions,[{
  key:hostKey,coordinate:[0,0,0],spanBricks:4,resolution:1,
  density:new Float64Array([1]),gamma:new Float64Array([1]),
 },{key,coordinate,unclipped:true,resolution:8,
  density:new Float64Array(512),gamma:new Float64Array(512).fill(1),
 }],0,8,true,false);
 const plan=planSparseCM12ResidentGeneration(atlas,new Set([hostKey]),new Map([
  [hostKey,{resolution:1 as const,mergeable:false,frozen:true}],
  [key,{resolution:8 as const,mergeable:false,activate:true}],
 ]),limits);
 assert.equal(plan?.status,"ready");if(plan?.status!=="ready")return;
 const host=plan.atlas.directory.get(hostKey)!;
 assert.equal(host.spanBricks,4);assert.equal(host.resolution,1);
 const receiver=plan.atlas.bricks.find(b=>b.key!==hostKey)!;
 assert.deepEqual(receiver.coordinate,[4,0,0]);assert.equal(receiver.spanBricks,2);
 assert.equal(receiver.resolution,1);assert.equal(receiver.unclipped,true);
 assert.equal(plan.active.size,2);assert.ok(plan.active.has(receiver.key));
 assert.ok(receiver.density.every(rho=>rho===0));
 assert.deepEqual(plan.newAirCoverage,[{minimumFine:[32,0,0],maximumExclusiveFine:[48,16,16]}]);
 assert.equal(atlas.bricks[1]!.spanBricks,undefined,"the source atlas remains immutable");
});

test("macro frontier growth refuses to absorb an existing frozen fine leaf", () => {
 const key=(q:readonly [number,number,number])=>sparseAtlasBrickKey(q,
  {brickDimensions:[4,4,4],signedCoordinates:true});
 const atlas=createSparseAdaptiveMassAtlas([32,32,32],[{
  key:key([0,0,0]),coordinate:[0,0,0],spanBricks:4,resolution:1,
  density:new Float64Array([1]),gamma:new Float64Array([1]),
 },...([[4,0,0],[5,1,1]] as const).map(coordinate=>({key:key(coordinate),coordinate,
  unclipped:true,resolution:8 as const,density:new Float64Array(512),gamma:new Float64Array(512).fill(1)}))],0,8,true,false);
 const active=new Set([key([0,0,0]),key([5,1,1])]);
 const intents=new Map(atlas.bricks.map(b=>[b.key,{resolution:b.resolution,mergeable:false,
  frozen:active.has(b.key),activate:!active.has(b.key)}]));
 assert.throws(()=>planSparseCM12ResidentGeneration(atlas,active,intents,limits),
  /would overlap accepted brick 5,1,1/);
});
test("frozen accepted cells constrain new support without rerunging the source", () => {
 const atlas=createSparseAdaptiveMassAtlas([16,8,8],[{
  key:0,coordinate:[0,0,0],resolution:1,
  density:new Float64Array([1]),gamma:new Float64Array([1]),
 },{key:1,coordinate:[1,0,0],resolution:8,
  density:new Float64Array(512),gamma:new Float64Array(512).fill(1),
 }],0,8,false,false);
 const plan=planSparseCM12ResidentGeneration(atlas,new Set([0]),new Map([
  [0,{resolution:8 as const,mergeable:true,frozen:true}],
  [1,{resolution:8 as const,mergeable:false,activate:true}],
 ]),limits);
 assert.equal(plan?.status,"ready");
 if(plan?.status!=="ready")return;
 assert.equal(plan.atlas.directory.get(0)!.resolution,1);
 assert.equal(plan.atlas.directory.get(1)!.resolution,2);
 assert.deepEqual([...plan.active],[0,1]);
 assert.ok(plan.atlas.directory.get(1)!.density.every(rho=>rho===0));
});

test("a frozen frontier activation needs publication even when every cell size matches", () => {
 const atlas=createSparseAdaptiveMassAtlas([16,8,8],[0,1].map(x=>({
  key:x,coordinate:[x,0,0] as const,resolution:8 as const,
  density:new Float64Array(512).fill(x===0?1:0),gamma:new Float64Array(512).fill(1),
 })),0,8,false,false);
 const plan=planSparseCM12ResidentGeneration(atlas,new Set([0]),new Map([
  [0,{resolution:8 as const,mergeable:false,frozen:true}],
  [1,{resolution:8 as const,mergeable:false,activate:true}],
 ]),limits);
 assert.equal(plan?.status,"ready");
 if(plan?.status!=="ready")return;
 assert.deepEqual([...plan.active],[0,1]);
 assert.ok(plan.atlas.bricks.every(b=>b.resolution===8));
});
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

test("thin-feature veto survives forced macro merging and region grading", () => {
 const bricks = Array.from({length:8},(_,key)=>({key,
  coordinate:[key&1,(key>>1)&1,(key>>2)&1] as const,
  resolution:1 as const,density:new Float64Array([1]),gamma:new Float64Array([1])}));
 const atlas=createSparseAdaptiveMassAtlas([16,16,16],bricks,0,8,false,false);
 const active=new Set(bricks.map(b=>b.key));
 const intents=new Map(bricks.map(b=>[b.key,{resolution:1 as const,mergeable:true,
  minimumCellWidth:16,protectThinFeatures:b.key===0}]));
 assert.equal(planSparseCM12ResidentGeneration(atlas,active,intents,limits)?.status,"deferred");
 const ordinary=new Map(bricks.map(b=>[b.key,{resolution:1 as const,mergeable:true,
  protectThinFeatures:b.key===0}]));
 assert.equal(planSparseCM12ResidentGeneration(atlas,active,ordinary,limits)?.status,"deferred");
});
