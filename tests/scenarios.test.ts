import assert from "node:assert/strict";
import test from "node:test";
import {validateScene} from "../lib/model";
import {createScenarioScene,SCENARIOS} from "../lib/scenarios";
import {EulerianFluidSolver} from "../lib/eulerian-solver";
import {createGPUHierarchyLayout} from "../lib/webgpu-hierarchy-layout";

test("H12-01 all showcase scenarios satisfy the serialized scene contract",()=>{
  for(const preset of SCENARIOS)assert.deepEqual(validateScene(createScenarioScene(preset.id)),[],preset.id);
});

test("H12-02 showcase scenarios exercise distinct hierarchy and coupling regimes",()=>{
  const wave=createScenarioScene("wave-tank"),buoyancy=createScenarioScene("buoyancy"),impact=createScenarioScene("splash-impact"),still=createScenarioScene("hydrostatic");
  assert.ok(wave.rigidBodies[0].linearVelocity_m_s.x>1);
  assert.ok(buoyancy.rigidBodies.some((body)=>body.density_kg_m3<buoyancy.fluid.density_kg_m3)&&buoyancy.rigidBodies.some((body)=>body.density_kg_m3>buoyancy.fluid.density_kg_m3));
  assert.ok(impact.rigidBodies[0].linearVelocity_m_s.y<0&&impact.hierarchy.regridInterval<wave.hierarchy.regridInterval);
  assert.equal(still.rigidBodies.length,0);assert.equal(still.fluid.initialCondition,"tank-fill");
});

test("H12-03 every scenario initializes finite CPU and hierarchical GPU state",()=>{
  for(const preset of SCENARIOS){
    const scene=createScenarioScene(preset.id),oracle=new EulerianFluidSolver(scene),step=oracle.step(scene.numerics.fixedDt_s),layout=createGPUHierarchyLayout(scene,"balanced");
    assert.equal(step.nanCount,0,preset.id);assert.ok(step.markerVolume_m3>0,preset.id);assert.ok(layout.activeCellCount>0,preset.id);assert.ok(layout.initialCells.every(Number.isFinite),preset.id);
  }
});

test("H12-04 visible free surfaces retain a finest-level predictive band",()=>{
  for(const preset of SCENARIOS){const scene=createScenarioScene(preset.id);assert.ok(scene.hierarchy.interfaceHaloCells>=scene.hierarchy.brickSize,preset.id);assert.ok(scene.hierarchy.minimumFluidLevel>=1,preset.id);}
  const dam=createGPUHierarchyLayout(createScenarioScene("dam-break"),"balanced"),ratio=dam.activeCellCount/dam.equivalentUniformCells;
  assert.ok(ratio>=.5,"surface band must not be starved for cells");assert.ok(ratio<.7,"hierarchy should retain a useful initial compression advantage");
});
