import assert from 'node:assert/strict';
import test from 'node:test';
import {paceDawnEventPolling} from '../lib/harness/node-dawn-provider';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {advanceUniform,createUniformSolver,readUniformFields,withUniformDevice} from './helpers/uniform-geometric';

test('symmetric ladder preserves initial contact and refinement and repeats through 60 steps',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 paceDawnEventPolling();
 await withUniformDevice('NB-FLIP initial symmetry',async device=>{
  const scene=sceneDocument(getSceneDefinition('sparse-cm12-ladder-symmetric-2d'));
  scene.numerics.fixedDt_s=1/60;
  const dims=[scene.container.width_m,scene.container.height_m,scene.container.depth_m].map(v=>Math.round(v/scene.voxelDomain.finestCellSize_m));
  const reference:Uint32Array[]=[];
  for(let run=0;run<2;run++){
   const solver=await createUniformSolver(device,structuredClone(scene));
   try{
    for(let step=1;step<=60;step++){
     await advanceUniform(solver,step/60);
     const fields=await readUniformFields(device,solver);
     let max=0;
     for(let z=0;z<=dims[2]!;z++)for(let y=0;y<=dims[1]!;y++)for(let x=0;x<=dims[0]!;x++)
      max=Math.max(max,Math.abs(fields.vertex(x,y,z)-fields.vertex(x,y,dims[2]!-z)));
     assert.ok(Number.isFinite(max),`run ${run}, step ${step}: finite surface`);
     // The first refinement used to introduce a 6 mm wall discrepancy.
     // Later spatial drift has a separate diagnostic; repeatability is
     // checked here through particle activation and the later impact.
     if(step<=36)assert.ok(max<(step<=8?1e-6:.002),`run ${run}, step ${step}: reflected surface error ${max} m`);
     const bits=new Uint32Array(fields.phi.buffer,fields.phi.byteOffset,fields.phi.length);
     if(run)assert.deepEqual(bits,reference[step-1],`step ${step}: repeatability`);else reference.push(bits.slice());
    }
   }finally{solver.destroy();}
  }
 },['subgroups']);
});
