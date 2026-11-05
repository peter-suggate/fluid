import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {wetClothSource,wetClothWaterSources,wetClothVariantSources,sourceClothDocument} from '../lib/cloth/scenes/sources';
import {prepareWetClothSource,buildWetClothSourceScene,wetClothSourceDefinitions,wetClothValidationDefinitions} from '../lib/cloth/scenes/definitions';
import {cachedClothTopology} from '../lib/cloth/source-assets';
import {parseScene,serializeScene,validateScene} from '../lib/core/model';
import {prepareClothTopology,createClothState} from '../lib/cloth/topology';
const request:typeof fetch=async url=>new Response(await readFile(`public${String(url)}`));

test('paper Figure10 parameter variants retain original source provenance and restore distinct physical parameters',async()=>{
 assert.equal(wetClothWaterSources.length,36);assert.equal(wetClothSourceDefinitions.length,36);assert.equal(wetClothVariantSources.length,3);assert.equal(wetClothValidationDefinitions.length,39);
 for(const id of ['wetcloth-paper-figure10b-25um','wetcloth-paper-figure10j-90deg']){
  const variant=wetClothSource(id),base=wetClothSource(variant.variant!.sourceId);
  assert.equal(variant.asset,base.asset);assert.equal(variant.source,base.source);assert.equal(variant.reproductionStatus,'paper-corrected-unverified');
  await prepareWetClothSource(id,request);const scene=buildWetClothSourceScene(id),document=scene.cloth![0]!;
  assert.equal(document.source!.configuration,'paper-corrected');assert.equal(document.source!.sha256,base.source.sha256);
  assert.deepEqual(validateScene(scene),[]);assert.deepEqual(parseScene(serializeScene(scene)).wetClothSource!.variant,variant.variant);
  assert.equal(scene.numerics.fixedDt_s,base.dt_s);assert.equal(scene.voxelDomain.finestCellSize_m,base.cellSize_m);
  if(id.endsWith('25um')){
   assert.equal(document.materials[0]!.fiberDiameter_m,25e-6);assert.equal(sourceClothDocument(base.id).materials[0]!.fiberDiameter_m,20e-6);
   const m=document.materials[0]!,radius=m.fiberDiameter_m/2*Math.sqrt((1-m.solidFraction)/m.solidFraction),threads=4*m.thickness_m*.0254*m.solidFraction/(Math.PI*m.fiberDiameter_m**2);
   assert.ok(Math.abs(radius*1e6-15.3093)<.0001);assert.ok(Math.abs(threads-7451.2)<.1,'supplement Table2 yields the Figure10b thread count');
  }else{
   assert.equal(document.materials[0]!.contactAngle_rad,Math.PI/2);assert.equal(document.materials[0]!.surfaceTension_N_m,.072);assert.equal(scene.fluid.surfaceTension_N_m,0,"source free-surface force stays disabled independently of capillary gamma");
   assert.equal(sourceClothDocument(base.id).materials[0]!.surfaceTension_N_m,0);
   assert.ok(Math.abs(Math.cos(document.materials[0]!.contactAngle_rad))<1e-15,'pore suction vanishes without zeroing water surface tension');
  }
 }
});

test('ring effective initialization preserves released zero capacity and exposes the material-packing hypothesis separately',async()=>{
 const base=wetClothSource('wetcloth-source-ring-test'),variant=wetClothSource('wetcloth-exploratory-ring-material-packing');
 await prepareWetClothSource(base.id,request);const original=cachedClothTopology(base.asset)!;
 assert.ok(original.solidFraction!.every(v=>v===1));assert.equal(base.sourceSettings.liquid.initNonuniformFraction,1);
 const rawDoc=sourceClothDocument(base.id),raw=createClothState(prepareClothTopology(original,rawDoc.materials));
 assert.ok(raw.solidFraction.every(v=>v===1));assert.equal(raw.controlVolume_m3.reduce((sum,v,i)=>sum+v*(1-raw.solidFraction[i]!),0),0);
 const scene=buildWetClothSourceScene(variant.id),doc=scene.cloth![0]!;assert.equal(doc.geometry.kind,'inline');if(doc.geometry.kind!=='inline')return;
 const state=createClothState(prepareClothTopology(doc.geometry.data,doc.materials));
 assert.ok(state.solidFraction.every(v=>v===.4));const capacity=state.controlVolume_m3.reduce((sum,v,i)=>sum+v*(1-state.solidFraction[i]!),0);
 assert.ok(Math.abs(capacity-4.2495357888e-6)<1e-12,'uniform material packing provides60% of source shell control volume');
 assert.equal(doc.source!.configuration,'interactive');assert.equal(scene.wetClothSource!.variant!.kind,'exploratory');assert.equal(variant.reproductionStatus,'exploratory-unverified');
 assert.ok(variant.variant!.reason.includes('not confirmed'));assert.deepEqual(validateScene(scene),[]);
 assert.ok(cachedClothTopology(base.asset)!.solidFraction!.every(v=>v===1),'exploratory input never mutates source asset');
 assert.equal(wetClothSource(base.id).sourceSettings.liquid.initNonuniformFraction,1);
 assert.deepEqual(parseScene(serializeScene(scene)).wetClothSource!.variant,variant.variant);
});
