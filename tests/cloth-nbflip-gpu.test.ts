import assert from 'node:assert/strict';
import test from 'node:test';
import {createRigidCouplingOracleScene} from '../lib/core/scenes';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState,prepareClothTopology} from '../lib/cloth/topology';
import {UniformClothGridCoupling,prepareUniformClothStencil} from '../lib/methods/uniform/uniform-cloth-grid-coupling';
import type {UniformMixedFrame} from '../lib/methods/uniform/uniform-mixed-frame';
import {uniformDetailField} from '../lib/methods/uniform/uniform-detail-fields';
import {createUniformSolver,withUniformDevice,readUniformFields} from './helpers/uniform-geometric';
import {UniformClothWaterExchange} from '../lib/methods/uniform/uniform-cloth-water-exchange';
import {readMixedBuffer} from './helpers/uniform-mixed-native-fields';
import {NB_VOLUME_COMPENSATION_FLOAT as compensation} from '../lib/methods/uniform/uniform-mixed-surface-volume';
import {clothPorousCapacityOf,clothHardAperture,clothCellWeights} from '../lib/methods/uniform/uniform-porous-capacity';
import {initializeRigidBodies} from '../lib/core/rigid-body';

test('actual NB-FLIP forced MAC exchange is reciprocal below the particle band', {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
 await withUniformDevice('cloth NB-FLIP exchange',async device=>{
  const scene=createRigidCouplingOracleScene('rigid-hydrostatic');scene.rigidBodies=[];
  scene.fluid.density_kg_m3=1000;
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=.001;
  scene.fluid.refinementRegions=[{id:'cloth-stencil',min_m:{x:-.2,y:.05,z:-.2},max_m:{x:.2,y:.3,z:.2},rule:'minimum-cell-size',minimumCellSize_cells:1,maximumCellSize_cells:1}];
  const solver=await createUniformSolver(device,scene);
  const frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame;
  const lattice=frame.ownership.capacity.lattice;
  const fields=frame.clothCouplingFields([lattice.origin_m.x,lattice.origin_m.y,lattice.origin_m.z],lattice.cellSize_m);
  const coupling=new UniformClothGridCoupling(fields.device,fields);
  // The local water exchange needs h owners on its whole derivative support
  // (MAC capture reservoir, its cell vertices and their dual-face neighbours:
  // three cells past each vertex cell). At +-.05 that support reached x=-.225,
  // outside the authored h region [-.2,.2] (a coarse 4h owner). +-.02 keeps it
  // (and the later +1 mm displacement) inside the h tiles, while the 4h owners
  // the phi query needs remain beyond.
  const data={restPositions_m:[-.02,.175,-.02,.02,.175,-.02,0,.175,.02],initialVelocities_m_s:[1,0,0,1,0,0,1,0,0],triangles:[0,1,2],rodOffsets:[0],rodIndices:[]};
  const state=createClothState(prepareClothTopology(data,[DEFAULT_CLOTH_MATERIAL]));
  try {
   await coupling.initialize();
   const step={state,data,materials:[DEFAULT_CLOTH_MATERIAL],identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001};
   const exchange=new UniformClothWaterExchange(fields);await exchange.initialize();await exchange.initializeCapacity(step);
   coupling.prepare(step);const stencil=prepareUniformClothStencil(fields,step);
   let volume:undefined|(()=>Promise<Float32Array>),before:undefined|(()=>Promise<Float32Array>),after:undefined|(()=>Promise<Float32Array>);
   frame.setPhysicsPlugins([{id:coupling.id,encodeSupport:e=>coupling.encodeSupport(e),encodeForces:(e,dt)=>{
    volume=uniformDetailField(fields.volume)!.storage.capture(e,fields.volume);
    before=uniformDetailField(fields.velocity)!.storage.capture(e,fields.velocity);
    coupling.encodeForces(e,dt);
    after=uniformDetailField(fields.velocity)!.storage.capture(e,fields.velocity);
   },encodeAfterProjection:e=>coupling.encodeAfterProjection(e)}]);
   assert.ok(solver.advanceTo(.001,[]));await solver.awaitFrameCompletion();
   assert.equal(solver.info.simulationPipelineError,undefined);
   const publication=await coupling.finish();assert.ok(before&&after);
   const [a,b,v]=await Promise.all([before(),after(),volume!()]);assert.equal(a.length,b.length);
   for(let axis=0;axis<3;axis++){
    let water=0,solid=0;
    const nx=solver.info.nx,ny=solver.info.ny;
    stencil.coordinates.forEach((p,i)=>{
     let first=stencil.anchors[8*i+3]!;for(let j=0;j<axis;j++)first+=stencil.anchors[8*i+4+j]!;
     let excluded=0;for(let j=0;j<stencil.anchors[8*i+4+axis]!;j++)excluded+=stencil.occupiedVolumes_m3[first+j]!;
     const at=4*(p[0]+nx*(p[1]+ny*p[2]))+axis;const q=[...p];q[axis]!++;const vi=p[0]+nx*(p[1]+ny*p[2]),vj=q[0]!+nx*(q[1]!+ny*q[2]!);water+=(b[at]!-a[at]!)*1000*fields.cellSize_m.reduce((x,y)=>x*y,1)*.5*(v[vi]!+v[vj]!);
    });
    for(let i=axis;i<publication.dragImpulse_Ns!.length;i+=3)solid+=publication.dragImpulse_Ns![i]!;
    assert.ok(Math.abs(water+solid)<1e-7,`axis ${axis} momentum defect ${water+solid}`);
    if(axis===0)assert.ok(water>1e-8&&solid< -1e-8,'moving cloth must push real water');
   }
   let pressureY=0;for(let i=1;i<publication.pressureImpulse_Ns!.length;i+=3)pressureY+=publication.pressureImpulse_Ns![i]!;
   const buoyancy=1000*state.solidVolume_m3.reduce((a,b)=>a+b,0)*-scene.fluid.gravity_m_s2.y*.001;
   assert.ok(Math.abs(pressureY-buoyancy)<.02*buoyancy,`projected pressure impulse ${pressureY} versus hydrostatic ${buoyancy}`);
   const sampledY=publication.fluidPressureAcceleration_m_s2!.filter((_,i)=>i%3===1);
   assert.ok(sampledY.every(value=>Math.abs(value+scene.fluid.gravity_m_s2.y)<.03*Math.abs(scene.fluid.gravity_m_s2.y)),'pressure samples must come from the hydrostatic native projection');
   coupling.prepare({...step,physics:{pressureOnSolid:false},identity:{...step.identity,tick:1,time_s:.001}});
   assert.ok(solver.advanceTo(.002,[]));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
   const ablation=await coupling.finish();assert.ok(ablation.pressureImpulse_Ns!.every(value=>value===0));
   assert.ok(ablation.fluidPressureAcceleration_m_s2!.filter((_,i)=>i%3===1).every(value=>value>9),'solid pressure ablation must retain the real free-liquid pressure sample');
   frame.setPhysicsPlugins([]);
   const native=await readUniformFields(device,solver),queries:number[]=[],expected:number[]=[],seen=new Set<number>();
   for(let z=1;z<solver.info.nz-1;z++)for(let y=1;y<solver.info.ny-1;y++)for(let x=1;x<solver.info.nx-1;x++){
    const width=native.widthAt(x,y,z);if(seen.has(width)||![1,4].includes(width))continue;seen.add(width);
    const q=[x+.31,y+.43,z+.67],origin=q.map(a=>Math.floor(a/width)*width),fraction=q.map((a,k)=>(a-origin[k]!)/width);let value=0;
    for(let c=0;c<8;c++){const corner=[c&1,(c>>1)&1,(c>>2)&1],weight=corner.reduce((w,b,a)=>w*(b?fraction[a]!:1-fraction[a]!),1);value+=weight*native.vertex(origin[0]!+width*corner[0]!,origin[1]!+width*corner[1]!,origin[2]!+width*corner[2]!);}
    queries.push(...q.map((a,k)=>fields.origin_m[k]!+a*fields.cellSize_m[k]!));expected.push(value);
   }
   assert.deepEqual([...seen].sort(),[1,4],'query fixture must contain actual fine and coarse accepted owners');
   const queried=await exchange.sampleLiquidPhi(queries);queried.forEach((value,i)=>assert.ok(Math.abs(value-expected[i]!)<Math.min(...fields.cellSize_m)*1e-5,`native mixed phi query ${i}: ${value} versus ${expected[i]}`));
   await assert.rejects(()=>exchange.sampleLiquidPhi([-3,-3,-3]),/no accepted resident liquid owner/);
   const beforeBudget=await readMixedBuffer(device,fields.waterBudget),cellVolume=fields.cellSize_m.reduce((a,b)=>a*b,1);
   const waterStep={...step,identity:{...step.identity,tick:1,time_s:.001}};
   const capturePhi=async()=>{const e=fields.device.createCommandEncoder(),read=uniformDetailField(fields.phi)!.storage.capture(e,fields.phi);fields.device.queue.submit([e.finish()]);return read();};
   const phiBefore=await capturePhi();
   const captured=await exchange.transfer(waterStep,Float64Array.from(state.controlVolume_m3,(v,i)=>.2*(v-state.solidVolume_m3[i]!)));
   assert.deepEqual(await capturePhi(),phiBefore,'submerged capture changes the material partition, not phi');
   assert.ok(captured.boundDelta_m3>0);assert.equal(captured.boundDelta_m3,-captured.freeDelta_m3);
   const afterBudget=await readMixedBuffer(device,fields.waterBudget);
   assert.ok(Math.abs(((afterBudget[0]!+afterBudget[compensation]!)-(beforeBudget[0]!+beforeBudget[compensation]!))*cellVolume-captured.freeDelta_m3)<1e-11);
   const released=await exchange.transfer({...waterStep,identity:{...waterStep.identity,tick:2}},Float64Array.from(state.waterVolume_m3,v=>-v));
   assert.ok(released.freeDelta_m3>0);assert.equal(released.boundDelta_m3,-released.freeDelta_m3);
   assert.ok(state.waterVolume_m3.reduce((a,b)=>a+b,0)<1e-10);
   for(let v=0;v<state.dryMass_kg.length;v++)state.positions_m[3*v]!+=.001;
   const beforeMove=await readMixedBuffer(device,fields.waterBudget),phiMove=await capturePhi();
   await exchange.reconcileCapacity({...waterStep,identity:{...waterStep.identity,tick:3}});
   const afterMove=await readMixedBuffer(device,fields.waterBudget);
   assert.equal(afterMove[0]!+afterMove[compensation]!,beforeMove[0]!+beforeMove[compensation]!);
   assert.deepEqual(await capturePhi(),phiMove,'submerged fiber displacement preserves surface phi');
   exchange.destroy();
  } finally {frame.setPhysicsPlugins([]);coupling.destroy();solver.destroy();}
 });
});

for(const clothY of [.175,.4])test(`moving rounded clamp preserves aperture-normalized fiber and the paired water ledger ${clothY===.4?'at the interface':'when submerged'}`,{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('cloth actual moving clamp apertures',async device=>{
  const scene=createRigidCouplingOracleScene('rigid-hydrostatic');scene.fluid.gravity_m_s2={x:0,y:0,z:0};
  scene.numerics.fixedDt_s=scene.numerics.maxDt_s=.001;
  scene.rigidBodies=[{...scene.rigidBodies[0]!,id:'moving-clamp',shape:'rounded-plate',dimensions_m:{x:.09,y:.015,z:.09},position_m:{x:0,y:clothY-.04,z:0},kinematic:true}];
  scene.fluid.refinementRegions=[{id:'clamp-material-support',min_m:{x:-.25,y:0,z:-.25},max_m:{x:.25,y:clothY+.15,z:.25},rule:'minimum-cell-size',minimumCellSize_cells:1,maximumCellSize_cells:1}];
  const solver=await createUniformSolver(device,scene),frame=(solver as unknown as {mixedFrame:UniformMixedFrame}).mixedFrame,lattice=frame.ownership.capacity.lattice;
  const fields=frame.clothCouplingFields([lattice.origin_m.x,lattice.origin_m.y,lattice.origin_m.z],lattice.cellSize_m),exchange=new UniformClothWaterExchange(fields);
  const data={restPositions_m:[-.05,clothY,-.05,.05,clothY,-.05,0,clothY,.05],triangles:[0,1,2],rodOffsets:[0],rodIndices:[]},state=createClothState(prepareClothTopology(data,[DEFAULT_CLOTH_MATERIAL]));
  const step={state,data,materials:[DEFAULT_CLOTH_MATERIAL],identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001};
  const bodies=initializeRigidBodies(scene.rigidBodies),solid=fields.solid as unknown as {bodySource:GPUBuffer;bodies:GPUBuffer};
  let clampCoupling:UniformClothGridCoupling|undefined;
  try{
   await exchange.initialize();await exchange.initializeCapacity(step);await frame.initializePhysicalWaterBudget();
   frame.requestNarrowBandVolumeProbe();assert.ok(solver.advanceTo(.001,bodies));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
   const phiSnapshot=async()=>{const e=fields.device.createCommandEncoder(),read=uniformDetailField(fields.phi)!.storage.capture(e,fields.phi);fields.device.queue.submit([e.finish()]);return read();};
   const initialPhi=await phiSnapshot(),initialLedger=await exchange.readFreeWaterVolume(),mirror=await readMixedBuffer(device,solid.bodies);
   const physicalVolume=async()=>{const measured=await readUniformFields(device,solver);return measured.density.reduce((a,b)=>a+b,0)*fields.cellSize_m.reduce((a,b)=>a*b,1);};
   console.log(JSON.stringify({clampY:clothY,phase:'before-motion',ledger:initialLedger,physical:await physicalVolume(),mirror:Array.from(mirror.slice(0,16))}));
   const beforeWeights=clothCellWeights(fields,state.positions_m.subarray(0,3));
   // Submerged, 11 mm: the faces at y=0.15 within |x|,|z|<0.09 then lie
   // strictly inside the moved plate. At 10 mm their outer corners touch its
   // rim exactly (sqrt(2)*0.01 = sqrt(r^2-0.005^2)), where f32 rounding alone
   // decides closure.
   const motion_m=clothY===.4?.015:.011,nextY=clothY-.04+motion_m;
   const moving=bodies.map(body=>({...body,position_m:{...body.position_m,y:nextY},linearVelocity_m_s:{x:0,y:motion_m/.001,z:0}}));
   fields.device.queue.writeBuffer(solid.bodySource,4,new Float32Array([nextY]));
   fields.device.queue.writeBuffer(solid.bodySource,12*4,new Float32Array([0,motion_m/.001,0]));
   const receipt=await exchange.reconcileCapacity({...step,identity:{...step.identity,tick:1,time_s:.001}},true,true);
   assert.equal(receipt.freeDelta_m3,0);assert.equal(receipt.boundDelta_m3,0);
   assert.deepEqual(await readMixedBuffer(device,solid.bodies),mirror,'sampling must not consume the previous native collision pose');
   assert.deepEqual(await phiSnapshot(),initialPhi,'aperture basis changes belong to native hard-domain motion, not an extra water source');
   assert.equal(await exchange.readFreeWaterVolume(),initialLedger);
   const afterWeights=clothCellWeights(fields,state.positions_m.subarray(0,3));
   console.log(JSON.stringify({clampY:clothY,phase:'aperture-refresh',beforeWeights,afterWeights}));
   assert.notDeepEqual(afterWeights,beforeWeights,'actual moving collider must change the shared interpolation weights');
   const table=clothPorousCapacityOf(fields.porousCapacity)!,fiber=table.cells.reduce((v,c)=>v+c.solidFraction*clothHardAperture(fields,c.p)*table.cellVolume_m3,0);
   assert.ok(Math.abs(fiber-state.solidVolume_m3.reduce((a,b)=>a+b,0))<1e-15,'hard aperture must not erase invariant fiber volume');
   for(let v=0;v<state.dryMass_kg.length;v++)assert.ok(Math.abs(clothCellWeights(fields,state.positions_m.subarray(3*v,3*v+3)).reduce((sum,c)=>sum+c.weight,0)-1)<1e-12);
   if(clothY===.175){
    clampCoupling=new UniformClothGridCoupling(fields.device,fields);await clampCoupling.initialize();clampCoupling.prepare({...step,identity:{...step.identity,tick:1,time_s:.001}});
    frame.setPhysicsPlugins([{id:clampCoupling.id,encodeSupport:e=>clampCoupling!.encodeSupport(e),encodeForces:(e,dt)=>clampCoupling!.encodeForces(e,dt),encodeAfterProjection:e=>clampCoupling!.encodeAfterProjection(e)}]);
   }
   frame.requestNarrowBandVolumeProbe();assert.ok(solver.advanceTo(.002,moving));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
   if(clampCoupling){
    await clampCoupling.finish();const stencil=prepareUniformClothStencil(fields,step),pairBuffer=(clampCoupling as unknown as {buffers:GPUBuffer[]}).buffers.find(b=>b.label==='Cloth pair impulses')!;
    const pairs=await readMixedBuffer(device,pairBuffer);let closedLinks=0;
    stencil.coordinates.forEach((p,i)=>{
     const faceY=fields.origin_m[1]+(p[1]+1)*fields.cellSize_m[1],x=fields.origin_m[0]+(p[0]+.5)*fields.cellSize_m[0],z=fields.origin_m[2]+(p[2]+.5)*fields.cellSize_m[2];
     if(Math.abs(faceY-.15)>1e-10||Math.abs(x)>=.09||Math.abs(z)>=.09)return;
     const first=stencil.anchors[8*i+3]!+stencil.anchors[8*i+4]!,count=stencil.anchors[8*i+5]!;
     for(let j=0;j<count;j++){assert.equal(pairs[first+j],0,'closed-face collider velocity cannot become cloth drag or pressure traction');closedLinks++;}
    });assert.ok(closedLinks>0,'fixture must exercise closed native MAC faces');
   }
   assert.equal(await exchange.readFreeWaterVolume(),initialLedger,'native collider motion retains the accepted free-water ledger');
   const cellVolume=fields.cellSize_m.reduce((a,b)=>a*b,1),history:{step:number;physical:number;phiVolume:number;budget:number[]}[]=[];
   const observe=async(step:number)=>{const budget=await readMixedBuffer(device,fields.waterBudget);const row={step,physical:await physicalVolume(),phiVolume:budget[4]!*cellVolume,budget:Array.from(budget)};history.push(row);console.log(JSON.stringify({clampY:clothY,phase:'after-motion',...row}));};
   await observe(2);
   const held=moving.map(body=>({...body,linearVelocity_m_s:{x:0,y:0,z:0}}));
   for(let tick=3;tick<=6;tick++){
    if(clampCoupling)clampCoupling.prepare({...step,physics:{solveSolid:false},identity:{...step.identity,tick:tick-1,time_s:(tick-1)*.001}});
    frame.requestNarrowBandVolumeProbe();assert.ok(solver.advanceTo(tick*.001,held));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);if(clampCoupling)await clampCoupling.finish();await observe(tick);
   }
   for(const sample of history)assert.ok(Math.abs(sample.phiVolume-initialLedger)<cellVolume*.01,`step ${sample.step}: native phi volume ${sample.phiVolume} disagrees with paired ledger ${initialLedger}; history=${JSON.stringify(history)}`);
  }finally{frame.setPhysicsPlugins([]);clampCoupling?.destroy();exchange.destroy();solver.destroy();}
 });
});
