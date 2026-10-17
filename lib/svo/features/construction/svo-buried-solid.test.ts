import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld } from "../../../core/solid-world";
import { buriedSolidClassifier } from "./svo-buried-solid";

const cell=[.05,.05,.05],origin=[0,0,0];
const edges=[[3.2,3.2,3.2],[1.6,1.6,1.6],[.8,.8,.8],[.4,.4,.4],[.2,.2,.2],[.1,.1,.1],[.05,.05,.05]];
const ground=createSolidWorld([{operation:"fill",minimum:[-32,-32,-32],maximumExclusive:[32,0,32],materialId:1}]);
const classify=(world=ground, blockers: Parameters<typeof buriedSolidClassifier>[6]=[])=>buriedSolidClassifier(world,origin,cell,origin,edges,8,blockers);

test("uniform source interiors and exterior stay coarse; curved blockers keep air refined",()=>{
  assert.equal(classify()(3,{x:0,y:-3,z:0}),true);
  assert.equal(classify()(3,{x:0,y:2,z:0}),true);
  assert.equal(classify(ground,[{minimum:[0,.8,0],maximum:[.4,1.2,.4]}])(3,{x:0,y:2,z:0}),false);
});
test("flat boundaries coarsen at most two levels and preserve edge detail",()=>{
  assert.equal(classify()(3,{x:0,y:0,z:0}),false);
  assert.equal(classify()(4,{x:0,y:0,z:0}),true);
  const box: NonNullable<Parameters<typeof buriedSolidClassifier>[6]>[number]={minimum:[-.5,.1,-.5],maximum:[.5,.11,.5],axisAlignedBox:true};
  assert.equal(classify(ground,[box])(4,{x:0,y:0,z:0}),false,"a thin slab must not disappear between coarse samples");
  const edge: NonNullable<Parameters<typeof buriedSolidClassifier>[6]>[number]={minimum:[0,-1,-1],maximum:[.4,.04,1],axisAlignedBox:true};
  assert.equal(classify(ground,[edge])(4,{x:0,y:0,z:0}),false,"box edges need fine cells");
});
test("cleared holes and non-monotone slabs prevent flat coarsening",()=>{
  const hole=createSolidWorld([...ground.patches,{operation:"clear",minimum:[1,-1,1],maximumExclusive:[2,0,2],materialId:0}]);
  assert.equal(classify(hole)(4,{x:0,y:0,z:0}),false);
  const slab=createSolidWorld([{operation:"fill",minimum:[-32,1,-32],maximumExclusive:[32,2,32],materialId:1}]);
  assert.equal(classify(slab)(4,{x:0,y:0,z:0}),false);
});

test("certified stone interiors coarsen while bevels, seams and suspended slabs stay fine",()=>{
  const stone = { minimum:[-.52,-.02,-.52], maximum:[.52,.05,.52],
    horizontalSlab:{minimum:[-.45,0,-.45],maximum:[.45,.03,.45]} } as const;
  assert.equal(classify(ground,[stone])(4,{x:0,y:0,z:0}),true);
  assert.equal(classify(ground,[stone])(4,{x:2,y:0,z:0}),false,"the inset excludes the bevel and joint");
  const raised = {...stone,minimum:[-.52,.08,-.52],maximum:[.52,.15,.52],
    horizontalSlab:{minimum:[-.45,.1,-.45],maximum:[.45,.13,.45]}} as const;
  assert.equal(classify(ground,[raised])(4,{x:0,y:0,z:0}),false,"a gap has multiple crossings");
});

test("native asset proxies coarsen by at most four while ordinary overlapping detail remains fine",()=>{
  const native={minimum:[-1,.8,-1],maximum:[1,2,1],nativeSurface:true} as const;
  assert.equal(classify(ground,[native])(3,{x:0,y:2,z:0}),false);
  assert.equal(classify(ground,[native])(4,{x:0,y:4,z:0}),true);
  const ordinary={minimum:[0,.8,0],maximum:[.4,1.2,.4]} as const;
  assert.equal(classify(ground,[native,ordinary])(4,{x:0,y:4,z:0}),false);
  const coarser=buriedSolidClassifier(ground,origin,cell,origin,edges.map(e=>e.map(v=>v*4)),8,[native]);
  assert.equal(coarser(4,{x:0,y:1,z:0}),false,"proxy cells never exceed 25 mm");
});
