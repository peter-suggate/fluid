import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeInitialPhi } from "../lib/methods/uniform/uniform-volume-initial";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";

test("uniform geometric vertex planes use metres and include all domain faces",()=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.fluid.initialCondition="tank-fill";scene.fluid.initialLiquidVolumes=[];scene.container.fillFraction=0.5;
  const phi=uniformVolumeInitialPhi(scene,[8,12,16]);assert.equal(phi.length,9*13*17);
  for(let z=0;z<=16;z++)for(let y=0;y<=12;y++)for(let x=0;x<=8;x++)assert.ok(Math.abs(phi[x+9*(y+13*z)]!-(y/12-0.5)*scene.container.height_m)<1e-6);
});
test("explicit shapes union with the reservoir; an empty scene has no zero-valued floor",()=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  scene.fluid.initialCondition="tank-fill";scene.container.fillFraction=0.25;
  scene.fluid.initialLiquidVolumes=[{shape:"sphere",center_m:{x:0,y:0.8*scene.container.height_m,z:0},radius_m:0.1}];
  const phi=uniformVolumeInitialPhi(scene,[10,10,10]);
  assert.ok(phi[5+11*(1+11*5)]!<0);assert.ok(phi[5+11*(8+11*5)]!<0);
  scene.container.fillFraction=0;scene.fluid.initialLiquidVolumes=[];
  assert.ok(uniformVolumeInitialPhi(scene,[8,8,8]).every(p=>p>0));
});
test("uniform geometric controls resolve independently from density algorithms",()=>{
  const values=resolveMethodValues(uniformVolumeMethod,"balanced",{});
  assert.equal(values.densitySharpening,"on");assert.equal(values.gammaDiffusion,undefined);
  assert.ok(uniformVolumeMethod.resolveComposition?.(values));
});
