import assert from "node:assert/strict";
import test from "node:test";
import {UNIFORM_MIXED_SOLID_ALL_FEATURES,uniformMixedSolidSceneFeatures,uniformMixedSolidWGSL} from "../lib/methods/uniform/uniform-mixed-solid.wgsl";

// Plain water compiles the solid library without the cloth plugins' porous
// table and without rounded-plate integration: no stage samples either.
test("plain water solid library compiles no porous table and no plate integration",()=>{
 const plain=uniformMixedSolidWGSL(2,undefined,true,false,{porous:false,plates:false});
 for(const name of ["umPorousWord","umPorousFreeCapacity","umSolidBodyCuts","umRigidIntegratedFraction","umRigidHasPlate","solidBoxCutDistance"])assert.equal(plain.includes(name),false,name);
 // The shared names resolve to the hard aperture, as before the plugins.
 assert.match(plain,/fn umFreeCellCapacity\(p:vec3i\)->f32\{return umCellOpen\(p\);\}/);
 assert.match(plain,/fn umApplyBodyCuts\(i:u32,local:vec3f,distance:f32\)->f32\{return distance;\}/);
 const all=uniformMixedSolidWGSL(2,undefined,true,false,UNIFORM_MIXED_SOLID_ALL_FEATURES);
 for(const name of ["umPorousWord","umPorousFreeCapacity","umSolidBodyCuts","umRigidIntegratedFraction"])assert.ok(all.includes(name),name);
});

test("scene features follow cloth, wet-cloth sources, a declared porous system, rounded plates and cuts",()=>{
 const box={shape:"box"} as const;
 assert.deepEqual(uniformMixedSolidSceneFeatures({rigidBodies:[box]}),{porous:false,plates:false});
 assert.deepEqual(uniformMixedSolidSceneFeatures({cloth:[{}],rigidBodies:[]}),{porous:true,plates:false});
 assert.deepEqual(uniformMixedSolidSceneFeatures({wetClothSource:{},rigidBodies:[]}),{porous:true,plates:false});
 assert.deepEqual(uniformMixedSolidSceneFeatures({rigidBodies:[box,{shape:"rounded-plate"}]}),{porous:false,plates:true});
 assert.deepEqual(uniformMixedSolidSceneFeatures({systems:{porous:true},rigidBodies:[]}),{porous:true,plates:false});
});
