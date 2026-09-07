import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { parseQueryState } from "../lib/core/url-state";
import { gpuSceneSeedKey } from "../lib/core/webgpu-renderer";
import { initializeSparseBrickAtlasFromScene } from "../lib/methods/adaptive-mass/sparse-brick-atlas";

import { createRerungFreeFallScene, createStandingWaveScene, standingWaveOmega } from "../lib/core/analytic-motion-scenes";
import { refinementKeyframeAt } from "../lib/core/refinement-regions";
import { initialHeightFieldFractionAtCell, initialHeightFieldRange } from "../lib/core/initial-height-field";

for (const motion of ["translation", "free-fall"] as const) test(`${motion} UI preset has exact box volume and a B1/B2 surface seam`, () => {
  const id = `coarse-surface-${motion}`, definition = getSceneDefinition(id);
  const scene = sceneDocument(definition);
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(serializeScene(scene)).fluid, scene.fluid);
  const query = parseQueryState(`?scene=${id}`);
  assert.equal(query.methodId, "adaptive-mass");
  assert.deepEqual(query.scene.fluid, scene.fluid);
  assert.equal(definition.methodProfile?.overrides.selectorMode, "coarse-first");
  const atlas = initializeSparseBrickAtlasFromScene(scene, { finestDimensions: [32,32,8],
    brickFineResolution: 8, maximumMacroSpanBricks: 1, coarseFirstCurvatureTolerance: .25 });
  const wet = atlas.bricks.filter(b => b.density.some(rho => rho > 0));
  assert.deepEqual(wet.map(b => [b.coordinate, b.resolution]), [[[1,2,0],1],[[2,2,0],2]]);
  const volume = wet.reduce((sum,b) => sum + b.density.reduce((a,v)=>a+v,0) * (.4/b.resolution)**3, 0);
  assert.ok(Math.abs(volume-.128)<1e-12);
  const t = scene.duration_s;
  assert.ok(.8 + scene.fluid.initialVelocity_m_s!.y*t + .5*scene.fluid.gravity_m_s2.y*t*t > .3,
    "analytic trajectory must finish before impact");
  const seedKey = gpuSceneSeedKey(scene);
  scene.fluid.initialVelocity_m_s!.y += .1;
  assert.notEqual(gpuSceneSeedKey(scene), seedKey, "editing starting velocity must reseed");
  scene.fluid.initialVelocity_m_s!.y = NaN;
  assert.ok(validateScene(scene).some(e => e.includes("Initial liquid velocity")));
});


test("free-fall keyframes refine and coarsen without changing the analytic body", () => {
  const scene=createRerungFreeFallScene();
  assert.deepEqual(validateScene(scene),[]);
  assert.equal(parseQueryState("?scene=coarse-surface-free-fall-rerung").methodId,"adaptive-mass");
  assert.deepEqual([0,.1,.2].map(t=>refinementKeyframeAt(scene,t)!.regions.map(r=>r.minimumCellSize_cells)),[[8,4],[4,8],[8,4]]);
  assert.equal(refinementKeyframeAt(scene,.099)!.time_s,0);
  scene.fluid.refinementKeyframes![1]!.time_s=0;
  assert.ok(validateScene(scene).some(e=>e.includes("strictly increasing")));
});

for(const live of [false,true]) test(`standing wave ${live ? "live" : "fixed"} preserves volume and has mixed surface resolution`,()=>{
  const scene=createStandingWaveScene(live),field=scene.fluid.initialHeightField!;
  assert.deepEqual(validateScene(scene),[]);
  assert.deepEqual(parseScene(serializeScene(scene)).fluid,scene.fluid);
  const query=parseQueryState(`?scene=${scene.sceneId}`);
  assert.equal(query.methodId,"adaptive-mass");
  const atlas=initializeSparseBrickAtlasFromScene(scene,{finestDimensions:[32,24,8],brickFineResolution:8,
    maximumMacroSpanBricks:1,coarseFirstCurvatureTolerance:.05});
  const surface=atlas.bricks.filter(b=>b.coordinate[1]===1&&b.density.some(v=>v>0));
  assert.deepEqual([...new Set(surface.map(b=>b.resolution))].sort(),[2,4]);
  let volume=0;
  for(let x=0;x<32;x++)for(let y=0;y<24;y++)for(let z=0;z<8;z++) volume+=initialHeightFieldFractionAtCell(scene,x,y,z,[32,24,8])!*.05**3;
  assert.ok(Math.abs(volume-.384)<1e-12,"cosine mode has exactly zero mean volume perturbation");
  assert.ok(Math.abs(2*Math.PI/standingWaveOmega()-1.574406504260834)<1e-12);
  assert.deepEqual(initialHeightFieldRange(field,-.8,.8,-.2,.2),[.57,.63]);
  const key=gpuSceneSeedKey(scene);
  assert.equal(field.kind,"cosine");
  if(field.kind==="cosine")field.amplitude_m=.02;
  assert.notEqual(gpuSceneSeedKey(scene),key,"editing wave amplitude must reseed the initial surface");
  if(field.kind==="cosine")field.wavelength_m=0;
  assert.ok(validateScene(scene).some(e=>e.includes("Cosine height field")));
});
