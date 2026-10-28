import assert from "node:assert/strict";
import test from "node:test";
import {createUniformTroughScene} from "../lib/core/uniform-trough-scenes";
import {createRigidCouplingOracleScene} from "../lib/core/scenes";
import {sceneLatticeDimensions} from "../lib/core/scene-lattice-dimensions";
import {bathInteriorContains,bathSolidContains} from "../lib/core/voxel-bath";
import {advanceUniform,createUniformSolver,readUniformFields,withUniformDevice} from "./helpers/uniform-geometric";

const gpu={skip:!process.env.WEBGPU_NODE_MODULE};

for(const mode of ["settled-tank","dam-break"] as const){
 test(`default ${mode} retains liquid inside voxel walls`,gpu,async t=>{
  await withUniformDevice(`voxel trough containment ${mode}`,async device=>{
   const scene=createUniformTroughScene(mode),h=scene.voxelDomain.finestCellSize_m;
   const solver=await createUniformSolver(device,scene);
   try{
    const {nx,ny}=solver.info,exterior:number[]=[];
    const [sx,sy,sz]=sceneLatticeDimensions(scene);
    // The top is open: the dam can run up and above its rim. Scan the
    // physical vessel below that rim, excluding the solver's solid padding.
    for(let z=0;z<sz;z++)for(let y=0;y<sy;y++)for(let x=0;x<sx;x++){
     const p=[(x+.5)*h-1.6,(y+.5)*h,(z+.5)*h-.6] as const;
     // Raw phi is deliberately continued inside solids. Check actual
     // volume in open exterior cells, not that solid-side continuation.
     if(!bathInteriorContains(...p)&&!bathSolidContains(...p))exterior.push(x+nx*(y+ny*z));
    }
    assert.ok(exterior.length>0);
    const steps=mode==="settled-tank"?120:60;
    let peakExterior=0;
    for(let step=0;step<=steps;step++){
     if(step)await advanceUniform(solver,step/60);
     if(step>2&&step%10!==0&&step!==steps)continue;
     const {density}=await readUniformFields(device,solver);
     let total=0,outside=0;
     for(const value of density){assert.ok(Number.isFinite(value));total+=value;}
     for(const i of exterior)outside+=density[i]!;
     assert.ok(total>1000,"containment must not pass by deleting the liquid");
     peakExterior=Math.max(peakExterior,outside);
     assert.ok(outside<1e-4,`frame ${step}: ${outside} cell volumes leaked into open exterior`);
    }
    t.diagnostic(`${steps} frames, peak exterior volume below the rim ${peakExterior*h**3} m³`);
   }finally{solver.destroy();}
  });
 });
}

test("a thin voxel partition retains its dry stencil at zero velocity, including legacy coarse settings",gpu,async()=>{
 await withUniformDevice("thin partition containment",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  scene.rigidBodies=[];
  scene.voxelDomain.finestCellSize_m=.025;
  scene.container.fillFraction=0;
  scene.fluid.gravity_m_s2={x:0,y:0,z:0};
  scene.fluid.dynamicViscosity_Pa_s=0;
  scene.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-.4,y:0,z:-.4},max_m:{x:-.025,y:.425,z:.4}}];
  // Two native cells, straddling a 4h boundary, with dry air beyond it.
  scene.solidVoxels=[{operation:"fill",minimum:[15,0,0],maximumExclusive:[17,32,32],materialId:1}];
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  const solver=await createUniformSolver(device,scene,{solidContact:"surface",warmResolution:"4h"});
  try{
   for(let step=1;step<=30;step++){
    await advanceUniform(solver,step/60);
    if(step>2&&step%10!==0)continue;
    const f=await readUniformFields(device,solver);let outside=0,inside=0;
    for(let z=0;z<32;z++)for(let y=0;y<32;y++)for(let x=0;x<32;x++){
     const v=f.density[x+32*(y+32*z)]!;assert.ok(Number.isFinite(v));
     if(x>=17)outside+=v;else if(x<15)inside+=v;
    }
    assert.ok(inside>1000);
    assert.ok(outside<1e-4,`frame ${step}: zero-velocity remap leaked ${outside} cell volumes through partition`);
    assert.equal(f.widthAt(14,4,16),1,"submerged wet cut tile retains native geometry");
    assert.equal(f.widthAt(17,4,16),1,"dry neighbouring cut tile retains the shared vertex stencil");
   }
  }finally{solver.destroy();}
 });
});
