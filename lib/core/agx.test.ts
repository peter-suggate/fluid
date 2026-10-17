import assert from "node:assert/strict";
import test from "node:test";
import { resolveDisplayGrade, sceneLinearToDisplay } from "./webgpu-lighting";
import { createBackyardPoolPartyScene } from "./backyard-pool-party-scene";
import { validateScene } from "./model";
test("AgX has neutral monotone greys and finite highlight compression",()=>{
  const grade=resolveDisplayGrade({toneCurve:"agx"});assert.equal(grade.toneCurve,"agx");
  let previous=0;
  for(const value of [0,1e-5,.01,.18,1,4,16,100]){
    const c=sceneLinearToDisplay([value,value,value],grade);
    assert.ok(c.every(v=>Number.isFinite(v)&&v>=0&&v<=1));
    assert.ok(Math.max(...c)-Math.min(...c)<.002);
    assert.ok(c[0]>=previous);previous=c[0];
  }
  const mid=sceneLinearToDisplay([.18,.18,.18],grade);assert.ok(mid[0]>.35&&mid[0]<.55);
  const fabric=sceneLinearToDisplay([.01,.4,.42],grade);
  assert.ok(fabric[0]>0&&fabric[1]>fabric[0],"bright fabric retains teal while compressing saturation");
  const scene=createBackyardPoolPartyScene();assert.equal(scene.lighting?.grade?.toneCurve,"agx");
  assert.deepEqual(validateScene(scene),[]);
});
