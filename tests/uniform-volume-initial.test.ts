import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeInitialPhi } from "../lib/methods/uniform/uniform-volume-initial";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { uniformMethod } from "../lib/methods/uniform/method";
import { UNIFORM_VOLUME_PIPELINE } from "../lib/methods/uniform/uniform-volume-pipeline";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { MethodParamValues } from "../lib/core/method-contract";
import type { FluidPipelineContext } from "../lib/core/fluid-pipeline";

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
test("the sharpening work map is the live default, with the dense schedule retained",()=>{
  const spec=uniformVolumeMethod.params.find(p=>p.key==="sharpeningWorkMap");
  assert.equal(spec?.kind,"select");assert.equal(spec?.default,"on");assert.equal(spec?.update,"runtime");
  assert.equal(resolveMethodValues(uniformVolumeMethod,"balanced",{}).sharpeningWorkMap,"on");
  // Runtime keys are excluded from the construction key, so the schedule toggles
  // on the attached solver instead of rebuilding it back to t=0.
  assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes("sharpeningWorkMap"));
  const stage=UNIFORM_VOLUME_PIPELINE.stages.find(s=>s.id==="uniform-volume-sharpen")!;
  const at=(values:MethodParamValues,info?:unknown):FluidPipelineContext=>({values,info:(info??null) as never,
    sceneId:"minimal-power-dam-break-32",bodyCount:0,hasTerrain:false,hasInflow:false,running:true});
  const counted={uniformSharpenWorkMap:true,uniformSharpenTilesActive:118,uniformSharpenTilesTotal:512};
  assert.equal(stage.chip(at({densitySharpening:"on",sharpeningWorkMap:"on"})),"4h work map");
  assert.equal(stage.chip(at({densitySharpening:"on",sharpeningWorkMap:"on"},counted)),"4h work map · 23% tiles");
  assert.equal(stage.chip(at({densitySharpening:"on",sharpeningWorkMap:"off"},counted)),"dense finest lattice");
  const choice=stage.controls?.find(c=>c.kind==="param-choice"&&c.param==="sharpeningWorkMap");
  assert.ok(choice?.kind==="param-choice");
  assert.equal(choice.enabled?.(at({densitySharpening:"on",sharpeningWorkMap:"on"})),true);
  assert.equal(choice.enabled?.(at({densitySharpening:"off",sharpeningWorkMap:"on"})),false);
  const readout=stage.controls?.find(c=>c.kind==="readout");
  assert.ok(readout?.kind==="readout");
  assert.equal(readout.value(at({sharpeningWorkMap:"on"},counted)),"118 / 512 (23%)");
  assert.equal(readout.value(at({sharpeningWorkMap:"on"})),"—");
  assert.equal(readout.value(at({sharpeningWorkMap:"off"},counted)),"—");
});
test("the volume dust floor is a live number surfaced on the transport stage",()=>{
  const spec=uniformVolumeMethod.params.find(p=>p.key==="volumeDustThreshold");
  assert.ok(spec?.kind==="number");assert.equal(spec.default,1e-6);assert.equal(spec.min,0);
  // Zero must remain reachable: it is the bit-identical control arm.
  assert.equal(resolveMethodValues(uniformVolumeMethod,"balanced",{volumeDustThreshold:0}).volumeDustThreshold,0);
  assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes("volumeDustThreshold"));
  assert.equal(uniformMethod.params.find(p=>p.key==="volumeDustThreshold"),undefined,"the paper method keeps the untreated sum");
  const stage=UNIFORM_VOLUME_PIPELINE.stages.find(s=>s.id==="uniform-volume-coupling")!;
  const at=(values:MethodParamValues,info?:unknown):FluidPipelineContext=>({values,info:(info??null) as never,
    sceneId:"minimal-power-dam-break-32",bodyCount:0,hasTerrain:false,hasInflow:false,running:true});
  assert.equal(stage.chip(at({volumeDustThreshold:0})),"dense finest lattice");
  assert.equal(stage.chip(at({volumeDustThreshold:1e-6})),"dust floor 1e-6");
  const slider=stage.controls?.find(c=>c.kind==="param-range"&&c.param==="volumeDustThreshold");
  assert.ok(slider&&slider.kind==="param-range");assert.equal(slider.min,0);
  const readout=stage.controls?.find(c=>c.kind==="readout");
  assert.ok(readout?.kind==="readout");
  const counted={uniformVolumeDustCells:412,uniformVolumeDustMass_cells:3.5e-4};
  assert.equal(readout.value(at({volumeDustThreshold:1e-6},counted)),"412 cells · 3.50e-4 cell volumes");
  assert.equal(readout.value(at({volumeDustThreshold:0},counted)),"—","off discards nothing to report");
  assert.equal(readout.value(at({volumeDustThreshold:1e-6})),"—");
});
test("the two-level velocity sampler is a live experiment on the velocity-extension stage",()=>{
  const toggle=uniformVolumeMethod.params.find(p=>p.key==="twoLevelVelocity");
  assert.equal(toggle?.kind,"select");assert.equal(toggle?.default,"on","the two-level path is the default; Off is the dense control");
  const reach=uniformVolumeMethod.params.find(p=>p.key==="twoLevelFineReach");
  assert.ok(reach?.kind==="number");assert.equal(reach.default,2);assert.equal(reach.min,0);assert.equal(reach.max,8);
  assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes("twoLevelVelocity"));
  assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes("twoLevelFineReach"));
  assert.equal(uniformMethod.params.find(p=>p.key==="twoLevelVelocity"),undefined,"the paper method stays all-fine");
  const stage=UNIFORM_VOLUME_PIPELINE.stages.find(s=>s.id==="velocity-extension")!;
  const at=(values:MethodParamValues,info?:unknown):FluidPipelineContext=>({values,info:(info??null) as never,
    sceneId:"minimal-power-dam-break-32",bodyCount:0,hasTerrain:false,hasInflow:false,running:true});
  const counted={uniformTwoLevelVelocity:true,uniformTwoLevelFineTiles:143,uniformTwoLevelTilesTotal:512,
    uniformTwoLevelShellTiles:205,uniformTwoLevelShellReach:1};
  assert.ok(!stage.chip(at({twoLevelVelocity:"off"}))?.includes("two-level"));
  assert.ok(stage.chip(at({twoLevelVelocity:"on"}))?.includes("two-level"));
  assert.ok(stage.chip(at({twoLevelVelocity:"on"},counted))?.includes("40%"),"the chip prices the shell, the set the passes run on");
  assert.ok(stage.chip(at({twoLevelVelocity:"on",twoLevelExtension:"dense"},counted))?.includes("28% fine"));
  // The extension stage's own sweep controls must survive the E1 additions.
  assert.ok(stage.controls?.some(c=>c.kind==="param-range"&&c.param==="extensionFrontSweeps"));
  const choice=stage.controls?.find(c=>c.kind==="param-choice"&&c.param==="twoLevelVelocity");
  assert.ok(choice?.kind==="param-choice");
  const slider=stage.controls?.find(c=>c.kind==="param-range"&&c.param==="twoLevelFineReach");
  assert.ok(slider?.kind==="param-range");
  assert.equal(slider.enabled?.(at({twoLevelVelocity:"on"})),true);
  assert.equal(slider.enabled?.(at({twoLevelVelocity:"off"})),false);
  const readout=stage.controls?.find(c=>c.kind==="readout"&&c.label==="Fine tiles");
  assert.ok(readout?.kind==="readout");
  assert.equal(readout.value(at({twoLevelVelocity:"on"},counted)),"143 / 512 (28%)");
  assert.equal(readout.value(at({twoLevelVelocity:"off"},counted)),"—");
});
test("E2 shrinks the extension and the far-air advection under the same sampler",()=>{
  for(const [key,fallback] of [["twoLevelExtension","dense"],["twoLevelAdvection","dense"]] as const){
    const spec=uniformVolumeMethod.params.find(p=>p.key===key);
    assert.equal(spec?.kind,"select");assert.equal(spec?.default,"tiles",`${key} ships shrunk`);
    assert.ok(spec?.kind==="select"&&spec.options.some(o=>o.value===fallback),"the dense control arm stays reachable");
    assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes(key),`${key} must flip live`);
    assert.equal(uniformMethod.params.find(p=>p.key===key),undefined,"the paper method keeps the dense schedule");
  }
  const reach=uniformVolumeMethod.params.find(p=>p.key==="twoLevelShellReach");
  assert.ok(reach?.kind==="number");assert.equal(reach.default,1);assert.equal(reach.max,8);
  assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes("twoLevelShellReach"));
  const extensionStage=UNIFORM_VOLUME_PIPELINE.stages.find(s=>s.id==="velocity-extension");
  assert.ok(extensionStage?.controls?.some(c=>c.kind==="param-range"&&c.param==="twoLevelShellReach"),"shell reach is on the panel");
  const at=(values:MethodParamValues,info?:unknown):FluidPipelineContext=>({values,info:(info??null) as never,
    sceneId:"minimal-power-dam-break-32",bodyCount:0,hasTerrain:false,hasInflow:false,running:true});
  const counted={uniformTwoLevelVelocity:true,uniformTwoLevelFineTiles:143,uniformTwoLevelTilesTotal:512,
    uniformTwoLevelShellTiles:205,uniformTwoLevelShellReach:1};
  const extension=UNIFORM_VOLUME_PIPELINE.stages.find(s=>s.id==="velocity-extension")!;
  const choice=extension.controls?.find(c=>c.kind==="param-choice"&&c.param==="twoLevelExtension");
  assert.ok(choice?.kind==="param-choice");
  assert.equal(choice.enabled?.(at({twoLevelVelocity:"on"})),true);
  assert.equal(choice.enabled?.(at({twoLevelVelocity:"off"})),false,"the shrink needs the sampler that replaces the field");
  const shell=extension.controls?.find(c=>c.kind==="readout"&&c.label==="Shell tiles");
  assert.ok(shell?.kind==="readout");
  assert.equal(shell.value(at({twoLevelVelocity:"on"},counted)),"205 / 512 (40%) · +1");
  assert.equal(shell.value(at({twoLevelVelocity:"off"},counted)),"—");
  // E2b sits on both stages it shrinks, never in a shelf away from the work.
  for(const id of ["velocity-advection","pressure-projection"]){
    const stage=UNIFORM_VOLUME_PIPELINE.stages.find(s=>s.id===id)!;
    assert.ok(stage,`${id} must survive the uniform-volume mapping`);
    const control=stage.controls?.find(c=>c.kind==="param-choice"&&c.param==="twoLevelAdvection");
    assert.ok(control?.kind==="param-choice",`${id} carries the far-air control`);
    assert.equal(control.enabled?.(at({twoLevelVelocity:"off"})),false);
    assert.ok(stage.chip(at({twoLevelVelocity:"on"},counted))?.includes("far air skipped"));
    assert.ok(!stage.chip(at({twoLevelVelocity:"on",twoLevelAdvection:"dense"},counted))?.includes("far air"));
    assert.ok(!stage.chip(at({twoLevelVelocity:"off"},counted))?.includes("far air"));
  }
});
test("extension front sweeps are a live budget surfaced on the velocity-extension stage",()=>{
  const spec=uniformVolumeMethod.params.find(p=>p.key==="extensionFrontSweeps");
  assert.equal(spec?.kind,"number");assert.equal(spec?.default,2);
  assert.equal(uniformMethod.params.find(p=>p.key==="extensionFrontSweeps")?.default,16,"paper method keeps the full budget");
  // A rebuild-class param would reset the simulation to t=0 on every slider tick.
  assert.ok(uniformVolumeMethod.runtimeParamKeys?.includes("extensionFrontSweeps"));
  const stage=UNIFORM_VOLUME_PIPELINE.stages.find(s=>s.id==="velocity-extension")!;
  const slider=stage.controls?.find(c=>c.kind==="param-range"&&c.param==="extensionFrontSweeps");
  assert.ok(slider&&slider.kind==="param-range");assert.equal(slider.min,1);assert.equal(slider.max,16);
  const readout=stage.controls?.find(c=>c.kind==="readout"&&c.label==="Sweeps with work");
  assert.ok(readout&&readout.kind==="readout");
  const at=(info:unknown):FluidPipelineContext=>({values:{extensionFrontSweeps:4},info:info as never,
    sceneId:"minimal-power-dam-break-32",bodyCount:0,hasTerrain:false,hasInflow:false,running:true});
  const facts={uniformPipelineFacts:{extrapolationFrontSweeps:4}};
  assert.equal(readout.value(at(null)),"—");
  assert.equal(readout.value(at({...facts,uniformFIMExecutedPasses:4,uniformFIMTerminalActiveFaces:37})),"4 of 4 · 37 faces unconverged");
  assert.equal(readout.value(at({...facts,uniformFIMExecutedPasses:3,uniformFIMTerminalActiveFaces:0})),"3 of 4 · converged");
});
