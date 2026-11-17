/** Physical bare-cloth assembly regression. Independent dense Float64 SPD/Galerkin
 * oracle; complements synthetic V-cycle tests. The contact case uses a real
 * detector-attached scene rather than invented inverse diagonals. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {clothGridData} from '../lib/cloth/grid';
import {prepareClothTopology} from '../lib/cloth/topology';
import {buildClothSolverModel} from '../lib/cloth/oracle';
import {ClothGPUSolver} from '../lib/cloth/gpu-solver';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
const material={id:'cloth',arealDensity_kg_m2:.2,stretchWarp_N_m:400,stretchWeft_N_m:250,shear_N_m:40,poisson:.3,compressionRatio:.05,bending_N_m:2e-5,damping_s:0,thickness_m:5e-4};
function cholesky(a:Float64Array,n:number){const L=new Float64Array(n*n);let smallest=Infinity;for(let i=0;i<n;i++)for(let j=0;j<=i;j++){let x=a[i*n+j]!;for(let k=0;k<j;k++)x-=L[i*n+k]!*L[j*n+k]!;if(i===j){assert.ok(Number.isFinite(x)&&x>0,`nonpositive physical MG pivot${i}: ${x}`);smallest=Math.min(smallest,x);L[i*n+j]=Math.sqrt(x);}else L[i*n+j]=x/L[j*n+j]!;}return smallest;}
test('physical mgAssemble is symmetric SPD, pins decouple, coarse blocks are Galerkin',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 try{for(const aggregate of [4,16])for(const mode of ['rest','stretch','compression','fold']){
  const data=clothGridData({columns:6,rows:6,width_m:.5,height_m:.5,origin_m:[0,1,0],group:(i,j)=>i===0&&j===0?1:0}),model=buildClothSolverModel(prepareClothTopology(data,[material]),[material],[]),positions=Float64Array.from(model.topology.data.restPositions_m);
  for(let v=1;v<36;v++){if(mode==='stretch')positions[3*v]*=1.3;if(mode==='compression')positions[3*v]*=.6;if(mode==='fold')positions[3*v+2]+=.04*Math.sin(v%6);}
  const solver=await ClothGPUSolver.create(device,model,{origin_m:[0,0,0],dt_s:1/60,gravity_m_s2:[0,-9.81,0],integrator:'bdf2',positions_m:positions,multigrid:true,multigridAggregateSize:aggregate,multigridFusedRows:2048,newton:{iterations:8,pcgIterations:100,pcgTolerance:1e-5,residualTolerance:1e-6,contactLinearization:'off'}});
  try{
   const mg=solver.tables.pcg.multigrid!,raw=solver as any,enc=device.createCommandEncoder(),pass=enc.beginComputePass(),run=(name:string,count:number,z=0)=>raw.run(pass,name,Math.ceil(count/64),z);
   run('beginStep',1);run('frame',solver.N);run('nwBeginStep',1);run('nwFaces',solver.T);run('nwGradient',solver.N);run('mgAssemble',mg.levels[0]!.blocks);for(let l=1;l<mg.levels.length;l++)run('mgCoarsen',mg.levels[l]!.blocks,l);pass.end();device.queue.submit([enc.finish()]);
   const work=new Float32Array(await raw.readRange(raw.newtonWork,0,16*solver.tables.newtonRecords)),topo=solver.tables.topo;let previous:Float64Array|undefined;
   for(let lev=0;lev<mg.levels.length;lev++){
    const l=mg.levels[lev]!,n=3*l.n,a=new Float64Array(n*n);
    for(let i=0;i<l.n;i++)for(let k=topo[l.rows+i]!;k<topo[l.rows+i+1]!;k++){const j=topo[l.columns+k]!;for(let r=0;r<3;r++)for(let c=0;c<3;c++)a[(3*i+r)*n+3*j+c]=work[4*(l.matrix+3*k+r)+c]!;}
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)assert.ok(Math.abs(a[i*n+j]!-a[j*n+i]!)<=2e-5*Math.max(1,Math.abs(a[i*n+j]!),Math.abs(a[j*n+i]!)),`matrix symmetry ${mode}/${lev}/${i}/${j}`);
    // Cholesky checks positive pivots, not merely positive determinant or random directions.
    try{cholesky(a,n);}catch(error){throw new Error(`aggregate${aggregate} ${mode} level${lev} n${n}: ${String(error)}`);}
    if(lev===0)for(let i=0;i<3;i++)for(let j=0;j<n;j++){assert.equal(a[i*n+j],i===j?1:0,'pinned row');assert.equal(a[j*n+i],i===j?1:0,'pinned column');}
    else{const fine=mg.levels[lev-1]!,nf=3*fine.n,expected=new Float64Array(n*n),magnitude=new Float64Array(n*n);for(let i=0;i<nf;i++)for(let j=0;j<nf;j++){const row=3*topo[fine.parent+Math.floor(i/3)]!+i%3,col=3*topo[fine.parent+Math.floor(j/3)]!+j%3,index=row*n+col,v=previous![i*nf+j]!;expected[index]+=v;magnitude[index]+=Math.abs(v);}for(let k=0;k<a.length;k++)assert.ok(Math.abs(a[k]!-expected[k]!)<=2e-5*Math.max(1,magnitude[k]!),`Galerkin ${mode}/${lev}/${k}`);}
    previous=a;
   }
  }finally{solver.destroy();}
 }await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);}finally{device.destroy();}
});

// Real detector-attached field/sample/friction contacts; no fabricated HINV records.
test('physical contact mgAssemble remains SPD on a draped sphere',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {ClothSystem,CLOTH_SCENE_STEP_S}=await import('../lib/cloth/system');
 const {bridson2002RotatingSphere}=await import('../lib/cloth/scenes/bridson-2002/index');
 const {bridson2002SceneDescription}=await import('../lib/cloth/scenes/bridson-2002/scene-description');
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups']:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),errors:string[]=[];
 device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 try{for(const aggregate of [4,16]){
  const scene=bridson2002SceneDescription(bridson2002RotatingSphere({nodesAcross:8}));
  const system=await ClothSystem.create(device,scene,{multigrid:true,multigridAggregateSize:aggregate,multigridFusedRows:2048,contactEntryCapacity:256,contactMandatoryRedoCap:64,statusInterval:1});
  try{
   await system.capture([20*CLOTH_SCENE_STEP_S]);assert.equal(system.failure,undefined,JSON.stringify(system.failure));
   const entry=(system as any).entries.values().next().value,solver=entry.solver as ClothGPUSolver,raw=solver as any,detector=system.contactFor(scene.cloth![0]!.id)!;
   const contacts=await detector.readContactEntries();assert.ok(contacts.counts.some(v=>v>0),'physical contact must actually be active');
   const mg=solver.tables.pcg.multigrid!,work=new Float32Array(await raw.readRange(raw.newtonWork,0,16*solver.tables.newtonRecords)),topo=solver.tables.topo;
   let previous:Float64Array|undefined;
   for(let lev=0;lev<mg.levels.length;lev++){
    const l=mg.levels[lev]!,n=3*l.n,a=new Float64Array(n*n);
    for(let i=0;i<l.n;i++)for(let k=topo[l.rows+i]!;k<topo[l.rows+i+1]!;k++){const j=topo[l.columns+k]!;for(let r=0;r<3;r++)for(let c=0;c<3;c++)a[(3*i+r)*n+3*j+c]=work[4*(l.matrix+3*k+r)+c]!;}
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)assert.ok(Math.abs(a[i*n+j]!-a[j*n+i]!)<=2e-5*Math.max(1,Math.abs(a[i*n+j]!),Math.abs(a[j*n+i]!)),`physical contact symmetry ${lev}/${i}/${j}`);
    cholesky(a,n);
    if(lev===0){for(let v=0;v<l.n;v++)if(solver.model.topology.groups[v]!>0)for(let r=0;r<3;r++)for(let j=0;j<n;j++){const i=3*v+r;assert.equal(a[i*n+j],i===j?1:0);assert.equal(a[j*n+i],i===j?1:0);}}
    else{const fine=mg.levels[lev-1]!,nf=3*fine.n,expected=new Float64Array(n*n),magnitude=new Float64Array(n*n);for(let i=0;i<nf;i++)for(let j=0;j<nf;j++){const row=3*topo[fine.parent+Math.floor(i/3)]!+i%3,col=3*topo[fine.parent+Math.floor(j/3)]!+j%3,index=row*n+col,v=previous![i*nf+j]!;expected[index]+=v;magnitude[index]+=Math.abs(v);}for(let k=0;k<a.length;k++)assert.ok(Math.abs(a[k]!-expected[k]!)<=2e-5*Math.max(1,magnitude[k]!),`physical contact Galerkin ${lev}/${k}`);}
    previous=a;
   }
  }finally{system.destroy();}
 }await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);}finally{device.destroy();}
});
