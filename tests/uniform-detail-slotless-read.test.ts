import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {createUniformSolver,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

// A coarse start compacts the h stores at construction: a block with no h
// tile holds no slot, and its values are its base's (the accessors' slotless
// load). The logical readback must read them the same way, not as a missing
// texel (it read NaN for every non-canonical texel of a slotless block).
test("a slotless block reads back as its base after a coarse start compacts the h stores",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("slotless readback",async device=>{
  for(const start of ["4h","h"] as const){
   const scene=sceneDocument(getSceneDefinition("hydrostatic-power-large-offset"));
   scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
   scene.fluid.tiers={...scene.fluid.tiers,start};
   const solver=await createUniformSolver(device,scene);
   try{
    const f=await readUniformFields(device,solver),v=await readMixedTexture(device,solver.velocityTexture);
    let phi=0,speed=0;for(const x of f.phi)if(!Number.isFinite(x))phi++;for(let i=0;i<v.length;i++)if(i%4!==3&&!Number.isFinite(v[i]!))speed++;
    assert.deepEqual({phi,speed},{phi:0,speed:0},`start ${start}: non-finite texels read back at construction`);
    // The waterline sits at the same height whichever start holds it: 4h's trilinear phi is exact for a plane.
    for(let z=0;z<=16;z+=4)for(let x=0;x<=32;x+=4){const a=f.vertex(x,15,z),b=f.vertex(x,16,z);
     assert.ok(a<0&&b>0,`start ${start}: waterline at ${[x,z]}`);assert.ok(Math.abs(0.05*(15+a/(a-b))-.7625)<1e-4,`start ${start}: waterline height at ${[x,z]}`);}
   }finally{solver.destroy();}
  }
 });
});
