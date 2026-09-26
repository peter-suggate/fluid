import type {MethodParamValues} from "../lib/core/method-contract";
import assert from "node:assert/strict";
import test from "node:test";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { resolveUniformGeometricValues, UNIFORM_GEOMETRIC_NATIVE_PARAMS } from "../lib/methods/uniform/uniform-geometric-parameters";

test("Uniform production resolution has no preparation or activation parameters", () => {
  for(const values of [{}, {prepareCoarseSimulation:"on",coarseSimulation:"on"}] as MethodParamValues[]){
    const resolved=resolveUniformGeometricValues(values);
    assert.equal("prepareCoarseSimulation" in resolved,false);
    assert.equal("coarseSimulation" in resolved,false);
    assert.equal(uniformGeometricSolverOptions(values).prepareCoarseSimulation,undefined);
  }
  assert.ok(!uniformVolumeMethod.runtimeParamKeys?.includes("coarseSimulation"));
  assert.ok(!UNIFORM_GEOMETRIC_NATIVE_PARAMS.some(p=>p.key==="coarseSimulation"||p.key==="prepareCoarseSimulation"));
});

test("Uniform region authoring exposes only 1 and 4 and snaps to four-cell tiles", async () => {
  const { studioRegionSpaceForMethod, refinementRegionChoices, refinementRegionFromDrag, refinementRegionResizePolicy, refinementRegionsToQuery, refinementRegionsFromQuery } = await import('../lib/core/editor-refinement-region');
  const { sceneDocument } = await import('../lib/core/scene-definition');
  const { getSceneDefinition } = await import('../lib/core/scenes');
  const { refinementRegionLattice } = await import('../lib/core/refinement-regions');
  const scene=sceneDocument(getSceneDefinition('minimal-power-dam-break-64'));
  const space=studioRegionSpaceForMethod('uniform-volume');
  assert.deepEqual(space.cellSizes,[1,4]);assert.equal(space.brick_cells,4);
  assert.deepEqual(studioRegionSpaceForMethod('adaptive-volume').cellSizes,[1,2,4,8,16,32]);
  const lattice=refinementRegionLattice(scene), h=lattice.cellSize_m[0]!;
  const region=refinementRegionFromDrag(scene,{x:lattice.origin_m.x,y:0,z:lattice.origin_m.z},
    {x:lattice.origin_m.x+3*h,y:0,z:lattice.origin_m.z+3*h},{methodId:'uniform-volume',draft:{rule:'minimum-cell-size',cellSize_cells:1,holdAtOneTier:true}})!;
  assert.ok(region);assert.equal(region.minimumCellSize_cells,1);assert.equal(region.maximumCellSize_cells,1);
  assert.ok(Math.abs((region.max_m.x-region.min_m.x)/h-4)<1e-10);
  const restored=refinementRegionsFromQuery({...scene,fluid:{...scene.fluid,refinementRegions:[region]}},refinementRegionsToQuery({...scene,fluid:{...scene.fluid,refinementRegions:[region]}}),'uniform-volume')[0]!;
  assert.ok(Math.abs(restored.max_m.x-region.max_m.x)<1e-12,"four-cell region survives URL round trip");
  const choices=refinementRegionChoices(scene,region,'uniform-volume');
  for(const id of ['minimumCellSize','maximumCellSize'])assert.deepEqual(choices.find(g=>g.id===id)!.options.map(o=>o.id),['1','4']);
  assert.deepEqual(refinementRegionResizePolicy(scene,region,'uniform-volume').snap_m,lattice.cellSize_m.map(v=>v*4));
});

test("saved prototype mixed settings cannot select the retired solver or discard native controls", () => {
  const values = resolveUniformGeometricValues({ prepareMixedSimulation: "on", mixedSimulation: "on", sharpeningStrength: .375 });
  assert.equal('mixedSimulation' in values, false);
  assert.equal('prepareMixedSimulation' in values, false);
  assert.equal("coarseSimulation" in values, false);
  const options = uniformGeometricSolverOptions(values);
  assert.equal('prepareMixedSimulation' in options, false);
  assert.equal(options.prepareCoarseSimulation, undefined);
  assert.equal(options.geometricVolume, true);
  assert.equal(options.sharpeningStrength, .375);
});
