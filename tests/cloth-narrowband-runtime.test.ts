import test from 'node:test';
import assert from 'node:assert/strict';
import { defaultScene, type SceneDescription } from '../lib/core/model';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { withUniformDevice, createUniformSolver, readUniformLiveAddress } from './helpers/uniform-geometric';
import {readMixedTileWords,readMixedBuffer} from './helpers/uniform-mixed-native-fields';
import {uniformTileKey} from '../lib/methods/uniform/uniform-lattice-address';

function scene():SceneDescription {
 const s=structuredClone(defaultScene);
 s.container={width_m:.32,height_m:.32,depth_m:.32,fillFraction:.75,top:'open',fluidWallMode:'free-slip'};
 s.voxelDomain={finestCellSize_m:.01,brickSize_cells:8};s.nominalResolution={length_m:.01};
 s.solidVoxels=[];s.rigidBodies=[];delete s.scenery;delete s.terrain;delete s.solidBoundary;
 s.fluid={density_kg_m3:1000,dynamicViscosity_Pa_s:0,surfaceTension_N_m:0,gravity_m_s2:{x:0,y:0,z:0},initialCondition:'tank-fill'};
 s.numerics={...s.numerics,fixedDt_s:.0001,maxDt_s:.0001};
 s.cloth=[{schemaVersion:'1.0.0',id:'immersed-sheet',materials:[{...DEFAULT_CLOTH_MATERIAL,youngModulus_Pa:1e4,mechanicalDamping_s:0}],
 geometry:{kind:'inline',data:{restPositions_m:[-.04,.12,-.04,.04,.12,-.04,-.04,.12,.04,.04,.12,.04],
 initialVelocities_m_s:[.2,0,0,.2,0,0,.2,0,0,.2,0,0],triangles:[0,2,1,1,2,3],rodOffsets:[0],rodIndices:[]}}}];
 return s;
}

test('production NB-FLIP completes reciprocal immersed cloth frames in tank and free-space windows',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('cloth actual NB-FLIP runtime',async device=>{
  for(const freeSpace of [false,true]){
  const s=scene();if(freeSpace)s.container.boundary='free-space';
  const solver=await createUniformSolver(device,s);
  try{
   const initial=solver.clothSurfaces[0]!;assert.ok(initial);assert.equal(initial.tick,0);
   const initialBudget=await readMixedBuffer(device,(solver as unknown as {mixedFrame:{narrowBandVolumeBudget:GPUBuffer}}).mixedFrame.narrowBandVolumeBudget);
   assert.equal(initialBudget[1],1,'physical free-water budget must be captured before the first advance');
   assert.equal(solver.info.referenceLiquidVolume_cells,initialBudget[0]!+initialBudget[9]!);
   assert.ok(solver.advanceTo(.0001));assert.ok(solver.framePending);
   assert.equal(solver.advanceTo(.0002),false,'cloth state cannot be reused while its exchange is pending');
   await solver.awaitFrameCompletion();
   const current=solver.clothSurfaces[0]!;assert.equal(current.tick,1);assert.equal(current.time_s,.0001);
   const dx=current.positions_m[0]!-initial.positions_m[0]!;
   assert.ok(dx>0&&dx<.2*.0001,`Immersed cloth must transfer momentum to actual water; displacement ${dx}`);
   assert.equal(initial.positions_m[0],-.04,'accepted snapshots must be immutable');
   assert.equal(solver.info.completedTime_s,.0001);
   assert.ok(solver.advanceTo(.0002));await solver.awaitFrameCompletion();assert.equal(solver.clothSurfaces[0]!.tick,2);
   const stats=await solver.readStats(),bound=Array.from(solver.clothSurfaces[0]!.waterVolume_m3).reduce((a,b)=>a+b,0);
   const initialWater=(initialBudget[0]!+initialBudget[9]!)*.01**3;
   // The free target is one f32 GPU scalar (~24576 cells, ulp 2^-9 cells =
   // 1.95e-9 m^3); each accepted exchange rounds it once. 1e-9 m^3 was half
   // an ulp, below what the stored ledger can represent. Allow 4 ulps.
   const targetUlp_m3=2**(Math.floor(Math.log2(initialBudget[0]!+initialBudget[9]!))-23)*.01**3;
   assert.ok(Math.abs(stats.narrowBandTargetVolume_cells!*.01**3+bound-initialWater)<4*targetUlp_m3,'accepted free and bound water must retain the physical initial budget');
   if(freeSpace)assert.ok(stats.uniformMixedFineCapacity!>512,'physical support and the FLIP particle band need independent owner capacity');
  }catch(error){
   const host=solver as unknown as {mixedRegionTable():Uint32Array};
   const address=await readUniformLiveAddress(device,solver),words=await readMixedTileWords(device,solver);
   const t=uniformTileKey(address,2,2,2);
   console.error(JSON.stringify({clothFailure:String(error),regions:Array.from(host.mixedRegionTable()).slice(0,20),anchorTile:t,anchorTileWord:words[t],info:solver.info}));
   throw error;
  }finally{solver.destroy();}
  }
 });
});

for(const absorb of [false,true])test(`production coupled cloth and free liquid share variable-step free fall ${absorb?'while absorbing':'without absorption'}`,{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('actual cloth and NB-FLIP free fall',async device=>{
  let s=scene();s.container.boundary='free-space';s.fluid.gravity_m_s2={x:0,y:-9.80665,z:0};
  const document=s.cloth![0]!,data=document.geometry.kind==='inline'?document.geometry.data:undefined;assert.ok(data);
  s.cloth=[{...document,physics:{absorption:absorb,captureRate:100},geometry:{kind:'inline',data:{...data,initialVelocities_m_s:Array(data.restPositions_m.length).fill(0)}}}];
  const solver=await createUniformSolver(device,s);
  try{
   let time=0;for(const dt of [.0001,.0002,.00005]){
    s={...s,numerics:{...s.numerics,fixedDt_s:dt,maxDt_s:dt}};solver.applySceneUniforms(s);time+=dt;
    assert.ok(solver.advanceTo(time));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
    assert.equal(solver.clothVelocityClock,'accepted-step','both phases exchange momentum at the accepted integer time');
    const surface=solver.clothSurfaces[0]!,expected=-9.80665*time;assert.ok(surface.velocities_m_s);
    for(let v=0;v<surface.waterVolume_m3.length;v++){
     assert.ok(Math.abs(surface.velocities_m_s[3*v+1]!-expected)<Math.abs(expected)*.005,`vertex ${v}, t=${time}: free-fall velocity ${surface.velocities_m_s[3*v+1]} versus ${expected}`);
     assert.ok(Math.abs(surface.velocities_m_s[3*v]!)<Math.abs(expected)*.005&&Math.abs(surface.velocities_m_s[3*v+2]!)<Math.abs(expected)*.005);
    }
   }
   if(absorb)assert.ok(solver.clothSurfaces[0]!.waterVolume_m3.some(v=>v>0),'wetting gate must exercise actual paired water uptake');
  }finally{solver.destroy();}
 });
});

test('production hydrostatic neutral cloth remains balanced through coupled kick and wetting',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('actual hydrostatic neutral cloth',async device=>{
  const s=scene();s.fluid.gravity_m_s2={x:0,y:-9.80665,z:0};const document=s.cloth![0]!,data=document.geometry.kind==='inline'?document.geometry.data:undefined;assert.ok(data);
  s.cloth=[{...document,materials:document.materials.map(m=>({...m,density_kg_m3:1000})),physics:{captureRate:100},geometry:{kind:'inline',data:{...data,initialVelocities_m_s:Array(data.restPositions_m.length).fill(0)}}}];
  const solver=await createUniformSolver(device,s);
  try{for(let step=1;step<=2;step++){assert.ok(solver.advanceTo(step*.0001));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);
   const surface=solver.clothSurfaces[0]!;assert.ok(surface.velocities_m_s);assert.ok(surface.velocities_m_s.every(v=>Math.abs(v)<9.80665*.0001*.01),`hydrostatic accepted velocity=${Array.from(surface.velocities_m_s)}`);
  }assert.ok(solver.clothSurfaces[0]!.waterVolume_m3.some(v=>v>0));}finally{solver.destroy();}
 });
});

test('cloth fine support covers actual MAC stencils in centered free-space windows and initial admission',async()=>{
 const {WebGPUUniformReferenceSolver}=await import('../lib/methods/uniform/webgpu-uniform-reference');
 const {ClothRuntime}=await import('../lib/cloth/runtime');
 const {prepareUniformClothStencil}=await import('../lib/methods/uniform/uniform-cloth-grid-coupling');
 const {sceneFluidFrame,sceneFluidBoxPage,sceneFluidLatticeDimensions}=await import('../lib/core/scene-lattice-dimensions');
 const {uniformLatticeAddress,uniformBoxTileIndex}=await import('../lib/methods/uniform/uniform-lattice-address');
 const s=scene();s.container.boundary='free-space';
 const runtime=new ClothRuntime(s.cloth![0]!),frame=sceneFluidFrame(s);
 const address=uniformLatticeAddress(sceneFluidLatticeDimensions(s),true,{}, {boxPage:sceneFluidBoxPage(s),freeSpace:true});
 const host=Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype),{scene:s,clothRuntimes:[runtime],address}) as {mixedRegionTable():Uint32Array;addPhysicsSeedTiles(tiles:Set<number>):void;physicsFineSlotNeed():number};
 const table=host.mixedRegionTable(),seed=new Set<number>();host.addPhysicsSeedTiles(seed);assert.equal(table[0],1);assert.ok(seed.size>0);
 const stencil=prepareUniformClothStencil(frame,{state:runtime.state,data:runtime.prepared.data,materials:runtime.document.materials,identity:runtime.identity,dt_s:.0001});
 for(const anchor of stencil.coordinates){
  const tile=anchor.map(v=>Math.floor(v/4));
  assert.ok(tile.every((v,a)=>v>=table[4+a]!&&v<table[7+a]!),`world stencil ${anchor} outside physical region`);
  const local=tile.map((v,a)=>v-8*address.boxPage[a]!);
  assert.ok(seed.has(uniformBoxTileIndex(address,local[0]!,local[1]!,local[2]!)),`initial admission omitted ${anchor}`);
 }
 assert.ok(address.boxPage[0]>0&&table[4]!>=8*address.boxPage[0],'free-space support must retain its actual world-window placement');
 // The box contains one page, but these physical tiles require detail storage
 // in 18 surrounding pages. Those claimable pages cannot be clipped from
 // the slot reservation merely because the cloth itself is inside the box.
 assert.equal(host.physicsFineSlotNeed(),18);
 for(let i=0;i<runtime.state.positions_m.length;i+=3)runtime.state.positions_m[i]!+=.5;
 const moved=host.mixedRegionTable();
 assert.ok(moved[4]!>=8*(address.boxPage[0]+1),'moving physical support must leave the initial one-page box');
 let pins:readonly {x:number;y:number;z:number}[]=[];
 Object.assign(host,{pageResidency:{pins:[]},mixedBuilder:{setRegions(){}},mixedDynamic:{setRegions(){}},
  mixedFrame:{pagePool:{pin(p:typeof pins){pins=p;}},cold:{trigger(){}}},staleMixed(){}});
 (host as unknown as {prepareClothStep(dt:number):void}).prepareClothStep(.0001);
 assert.ok(pins.some(p=>p.x>address.boxPage[0]),'the native page owner must receive the moved footprint');
 runtime.dispose();
});

test('cloth native advance waits for current collider apertures before preparing adjoint force stencils',async()=>{
 const {WebGPUUniformReferenceSolver}=await import('../lib/methods/uniform/webgpu-uniform-reference');
 const {ClothRuntime}=await import('../lib/cloth/runtime');
 const runtime=new ClothRuntime(scene().cloth![0]!),events:string[]=[];
 let release!:()=>void;const fence=new Promise<void>(resolve=>{release=resolve;});
 const host=Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype),{disposed:false,scene:scene(),clothRuntimes:[runtime],prepareClothCohesionPairs:async()=>[],
  clothWaterExchanges:[{async refreshApertures(){events.push('sample-current-collider');await fence;events.push('capacity-published');}}],
  clothCouplings:[{prepare(){events.push('force-stencil');}}],mixedKicked:true,
  mixedFrameTrace(){return undefined;},readMixedRelayout(){events.push('read-layout');},
 }) as {advanceClothFluidFrame(frame:unknown,parameters:unknown,dt:number,bodies:undefined,roster:readonly unknown[]):Promise<unknown>};
 Object.defineProperty(host,'mixedGpuLayout',{value:false});
 const receipt={accepted:true},frame={advance(){events.push('native-advance');return Promise.resolve(receipt);}};
 // A moving collider in the roster requires current-pose apertures.
 const completion=host.advanceClothFluidFrame(frame,{},.001,undefined,[{description:{id:'clamp'}}]);
 assert.deepEqual(events,['sample-current-collider']);release();
 assert.equal(await completion,receipt);
 assert.deepEqual(events,['sample-current-collider','capacity-published','force-stencil','native-advance','read-layout']);runtime.dispose();
});

test('dry cloth pages are initially resident before a later liquid arrival',async()=>{
 const {uniformPageSources}=await import('../lib/methods/uniform/webgpu-uniform-reference');
 const {uniformPagePoolResidency}=await import('../lib/methods/uniform/uniform-page-pool');
 const {sceneFluidBoxPage,sceneFluidLatticeDimensions}=await import('../lib/core/scene-lattice-dimensions');
 const s=scene();s.container.boundary='free-space';s.container.height_m=1.28;s.container.fillFraction=0;
 const data=s.cloth![0]!.geometry;assert.equal(data.kind,'inline');if(data.kind!=='inline')return;
 s.cloth=[{...s.cloth![0]!,geometry:{kind:'inline',data:{...data.data,restPositions_m:data.data.restPositions_m.map((p,i)=>i%3===1?p+.8:p)}}}];
 const cells=sceneFluidLatticeDimensions(s) as [number,number,number];
 const residency=uniformPagePoolResidency(cells,undefined,uniformPageSources(s,undefined,.0001),true,()=>new Float32Array(cells[0]*cells[1]*cells[2]),sceneFluidBoxPage(s));
 assert.ok(residency.stored.some(([,y])=>y===2),'dry cloth must seed its own page without any initial water');
 assert.ok(residency.pins.length>0);
});

test('a rejected native impulse receipt rolls back the staged cloth step',async()=>{
 const {WebGPUUniformReferenceSolver}=await import('../lib/methods/uniform/webgpu-uniform-reference');
 const {ClothRuntime}=await import('../lib/cloth/runtime');
 const runtime=new ClothRuntime(scene().cloth![0]!),checkpoint=runtime.checkpoint();
 runtime.prepareNonFluidPredictor(.001,{gravity_m_s2:[0,-9.81,0]});
 const host=Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype),{
  clothRuntimes:[runtime],clothStepCheckpoints:[checkpoint],clothStepCohesion:[[]],
  clothCouplings:[{async finish(){throw new Error('native receipt rejected');}}],
 }) as {finishClothStep(dt:number,bodies:unknown[]):Promise<void>;clothStepCheckpoints?:unknown};
 await assert.rejects(host.finishClothStep(.001,[]),/native receipt rejected/);
 assert.equal(host.clothStepCheckpoints,undefined);
 // A different dt is legal again only after the staged target was cleared.
 runtime.prepareNonFluidPredictor(.002,{gravity_m_s2:[0,0,0]});
 assert.equal(runtime.state.time_s,0);runtime.restore(checkpoint);runtime.dispose();
});

test('cloth startup uses the ordinary native frame without a staggered initialization kick',async()=>{
 const {WebGPUUniformReferenceSolver}=await import('../lib/methods/uniform/webgpu-uniform-reference');const events:string[]=[];
 const host=Object.assign(Object.create(WebGPUUniformReferenceSolver.prototype),{clothRuntimes:[{}],mixedKicked:false,async prepareClothFluidStep(){events.push('prepare');},async kickMixedFrame(){throw new Error('unpaired half-kick');},mixedFrameTrace(){},readMixedRelayout(){events.push('read');}}) as {advanceClothFluidFrame(frame:unknown,p:unknown,dt:number,bodies:undefined):Promise<unknown>;mixedKicked:boolean};
 Object.defineProperty(host,'mixedGpuLayout',{value:false});const receipt={accepted:true};const result=await host.advanceClothFluidFrame({advance(){events.push('actual-native-frame');return Promise.resolve(receipt);}}, {},.001,undefined);
 assert.equal(result,receipt);assert.equal(host.mixedKicked,true);assert.deepEqual(events,['prepare','actual-native-frame','read']);
});

test('accepted cloth momentum changes only by actual native impulses during transient drag',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('actual transient cloth momentum clock',async device=>{
  let s=scene();s.container.boundary='free-space';s.cloth=s.cloth!.map(d=>({...d,physics:{absorption:false}}));const solver=await createUniformSolver(device,s);
  const host=solver as unknown as {clothRuntimes:{state:{dryMass_kg:Float64Array;velocities_m_s:Float64Array}}[];clothCouplings:{finish():Promise<{impulse_Ns:ArrayLike<number>}>}[]};
  const state=host.clothRuntimes[0]!.state,mass=state.dryMass_kg.slice(),coupling=host.clothCouplings[0]!,finish=coupling.finish.bind(coupling);let impulse:ArrayLike<number>|undefined;
  coupling.finish=async()=>{const receipt=await finish();impulse=Float64Array.from(receipt.impulse_Ns);return receipt;};
  const momentum=()=>[0,1,2].map(a=>mass.reduce((sum,m,i)=>sum+m*state.velocities_m_s[3*i+a]!,0));
  try{let time=0;for(const dt of [.0001,.0002,.00005]){
   const before=momentum();impulse=undefined;s={...s,numerics:{...s.numerics,fixedDt_s:dt,maxDt_s:dt}};solver.applySceneUniforms(s);time+=dt;
   assert.ok(solver.advanceTo(time));await solver.awaitFrameCompletion();assert.equal(solver.info.simulationPipelineError,undefined);assert.ok(impulse);const after=momentum();
   for(let a=0;a<3;a++){const actual=mass.reduce((sum,_m,i)=>sum+impulse![3*i+a]!,0);assert.ok(Math.abs(after[a]!-before[a]!-actual)<1e-8,`axis ${a} unpaired transient momentum: accepted=${after[a]!-before[a]!}, native=${actual}`);}
  }}finally{solver.destroy();}
 });
});
