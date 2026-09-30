import assert from "node:assert/strict";
import test from "node:test";
import {uniformVolumeMethod} from "../uniform/uniform-volume-method";

const resolve=uniformVolumeMethod.resolveComposition!;
const params=uniformVolumeMethod.params;
const selected=(values:Record<string,string|boolean>,key:string)=>resolve(values).variants.find(v=>v.point===`simulation.uniform-volume.algorithms.${key}`);

test("Uniform Geometric algorithm choices compose pairwise with their declared lifecycles",()=>{
 for(const a of params)for(const b of params){
  if(a.kind!=="select"||b.kind!=="select"||a.key===b.key)continue;
  for(const av of a.options)for(const bv of b.options){
   const values: Record<string,string>={[a.key]:av.value,[b.key]:bv.value};
   assert.equal(selected(values,a.key)?.id,av.value);
   assert.equal(selected(values,b.key)?.id,bv.value);
   assert.equal(selected(values,a.key)?.update,a.update==="runtime"?"live":"rebuild");
  }
 }
});

test("Uniform Geometric rejects unsupported algorithm choices",()=>{
 assert.throws(()=>resolve({coarsening:"octree"}),/supported variant/);
 assert.equal(uniformVolumeMethod.id,"uniform-volume");
 assert.ok(!params.some(p=>["selectorMode","maximumLeafSize","globalFineLevelSetFactor"].includes(p.key)));
});

test("Uniform Geometric rejects boolean overrides for select parameters",()=>{
 for(const p of params){if(p.kind!=="select")continue;
  for(const value of [true,false])assert.throws(()=>selected({[p.key]:value},p.key),/supported variant/);
 }
});
