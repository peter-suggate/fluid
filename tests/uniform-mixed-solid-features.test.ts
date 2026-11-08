import assert from "node:assert/strict";
import test from "node:test";
import {UNIFORM_MIXED_SOLID_ALL_FEATURES,uniformMixedSolidSceneFeatures,uniformMixedSolidWGSL} from "../lib/methods/uniform/uniform-mixed-solid.wgsl";

// Plain water compiles the solid library without rounded-plate integration:
// no stage samples the body-cut table.
test("plain water solid library compiles no plate integration",()=>{
 const plain=uniformMixedSolidWGSL(2,undefined,true,false,{plates:false});
 for(const name of ["umPorous","umSolidBodyCuts","umRigidIntegratedFraction","umRigidHasPlate","solidBoxCutDistance"])assert.equal(plain.includes(name),false,name);
 // Free-water capacity is the hard aperture.
 assert.match(plain,/fn umFreeCellCapacity\(p:vec3i\)->f32\{return umCellOpen\(p\);\}/);
 assert.match(plain,/fn umApplyBodyCuts\(i:u32,local:vec3f,distance:f32\)->f32\{return distance;\}/);
 const all=uniformMixedSolidWGSL(2,undefined,true,false,UNIFORM_MIXED_SOLID_ALL_FEATURES);
 for(const name of ["umSolidBodyCuts","umRigidIntegratedFraction"])assert.ok(all.includes(name),name);
 assert.equal(all.includes("umPorous"),false);
});

test("scene features follow rounded plates and cuts",()=>{
 const box={shape:"box"} as const;
 assert.deepEqual(uniformMixedSolidSceneFeatures({rigidBodies:[box]}),{plates:false});
 assert.deepEqual(uniformMixedSolidSceneFeatures({rigidBodies:[box,{shape:"rounded-plate"}]}),{plates:true});
});
