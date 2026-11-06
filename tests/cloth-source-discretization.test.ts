import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {wetClothSource,wetClothWaterSources,wetClothUpstreamDiscretizationSources} from '../lib/cloth/scenes/sources';
import {prepareWetClothSource,buildWetClothSourceScene} from '../lib/cloth/scenes/definitions';
import {applyWetClothSourceDiscretization,wetClothSourceDiscretizations} from '../lib/cloth/scenes/discretization';
import {parseScene,serializeScene,validateScene} from '../lib/core/model';
const request:typeof fetch=async url=>new Response(await readFile(`public${String(url)}`));

test('capillary suction resolves its tube and drop at 8h by default and keeps the released dx as a variant',async()=>{
 const id='wetcloth-source-capillary-suction',source=wetClothSource(id),released=wetClothSource(`${id}-upstream-dx`),d=wetClothSourceDiscretizations.get(id)!;
 assert.equal(wetClothWaterSources.length,36);assert.deepEqual(wetClothUpstreamDiscretizationSources.map(s=>s.id),[`${id}-upstream-dx`]);
 assert.equal(source.cellSize_m,.00072);assert.equal(source.dt_s,.00025);assert.equal(source.discretization,d);
 assert.ok(source.auditNotes.some(n=>n.startsWith('Discretization: dx 0.072 cm')),'the deviation from the XML is recorded in the scene notes');
 // Courant number of the released pair.
 assert.ok(Math.abs(source.dt_s/source.cellSize_m-d.sourceDt_s/d.sourceCellSize_m)<1e-12);
 for(const f of source.fields.filter(f=>f.root||f.usage==='source')){
  const r=f.kind==='sphere'?f.radius_m:f.usage==='solid'?f.roundingRadius_m:undefined;
  if(r!==undefined)assert.ok(r/source.cellSize_m>=8-1e-9,`field ${f.index} (${f.usage} ${f.kind}) spans ${r/source.cellSize_m}h`);
 }
 assert.equal(released.cellSize_m,d.sourceCellSize_m);assert.equal(released.dt_s,d.sourceDt_s);assert.equal(released.discretization,undefined);
 assert.equal(released.variant!.kind,'exploratory');assert.equal(released.source,source.source);assert.equal(released.asset,source.asset);
 await prepareWetClothSource(id,request);
 for(const s of [source,released]){
  const scene=buildWetClothSourceScene(s.id);
  assert.equal(scene.voxelDomain!.finestCellSize_m,s.cellSize_m);assert.equal(scene.numerics.fixedDt_s,s.dt_s);assert.equal(scene.wetClothSource!.timeStep_s,s.dt_s);
  assert.deepEqual(validateScene(scene),[]);assert.deepEqual(parseScene(serializeScene(scene)).voxelDomain,scene.voxelDomain);
 }
 // The liquid and the tube are refined locally: refinement regions cover the
 // liquid, and the scene's box is far coarser than its finest cell.
 const scene=buildWetClothSourceScene(id);assert.ok(scene.fluid.refinementRegions?.length);
 assert.throws(()=>applyWetClothSourceDiscretization({...source,discretization:undefined}),/does not match the released dx and dt/);
});
