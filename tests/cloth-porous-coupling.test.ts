import assert from 'node:assert/strict';
import test from 'node:test';
import { DEFAULT_CLOTH_MATERIAL as material } from '../lib/cloth/material';
import { porousPermeability, porousSuctionPressure, poreCapacity_m3, porousTensor, porousDragTensor, quasiStaticPorousSpeed, porousCaptureFraction, porousDragCoefficients } from '../lib/cloth/porous-material';
import { diffusePoreWater, planPoreAbsorption } from '../lib/cloth/porous-transport';
import { implicitDragExchange, implicitTensorDragExchange, ClothImpulseConsumer } from '../lib/cloth/coupling';
import { createClothState, prepareClothTopology } from '../lib/cloth/topology';
import {prepareGridDragMaterial} from '../lib/methods/uniform/uniform-cloth-drag-material';
import { prepareUniformClothStencil } from '../lib/methods/uniform/uniform-cloth-grid-coupling';
import type { ClothTopologyData } from '../lib/cloth/types';
import { planLocalLevelSetExchange, planLocalCapacityRemap, planLocalPoreExchange, remapCapacityMomentum, tetrahedralWaterFill } from '../lib/cloth/grid-water-transfer';
import { mixturePressureFace } from '../lib/methods/uniform/uniform-pressure-mixture';
import {prepareClothPorousCapacity,clothCellWeights,clothMacCaptureReservoir,setClothHardApertures} from '../lib/methods/uniform/uniform-porous-capacity';
import { advanceClothPoreTransport,acceptPositiveManifoldFlux,manifoldForcePredictorSpeed } from '../lib/cloth/porous-flow';

export const dragTriangle: ClothTopologyData = { restPositions_m: [-.1,.2,-.1, .1,.2,-.1, 0,.2,.1], initialVelocities_m_s: [1,0,0,1,0,0,1,0,0], triangles:[0,1,2],rodOffsets:[0],rodIndices:[] };

test('porous constitutive law has the physical signs and diameter scaling', () => {
  const k=porousPermeability(.4,1e-4), twice=porousPermeability(.4,2e-4);
  assert.ok(k.alongFiber_m2>0&&k.acrossFiber_m2>0);
  assert.equal(twice.alongFiber_m2,4*k.alongFiber_m2);
  const suction=porousSuctionPressure(.4,material);
  assert.ok(suction.alongFiber_Pa>0); assert.equal(suction.alongFiber_Pa,2*suction.acrossFiber_Pa);
  assert.equal(porousSuctionPressure(.4,{...material,contactAngle_rad:Math.PI/2}).alongFiber_Pa,0);
  assert.ok(porousSuctionPressure(.4,{...material,contactAngle_rad:Math.PI}).alongFiber_Pa<0);
  assert.equal(poreCapacity_m3(1,.4),.6);
});

test('stiff reciprocal drag conserves momentum and dissipates kinetic energy', () => {
  for(const c of [1e-8,1,1e12]) {
    const r=implicitDragExchange(2,3,4,-2,c,.1);
    assert.equal(r.fluidImpulse_Ns,-r.solidImpulse_Ns);
    assert.ok(Math.abs(2*r.fluidVelocity_m_s+3*r.solidVelocity_m_s-2)<1e-12);
    assert.ok(2*r.fluidVelocity_m_s**2+3*r.solidVelocity_m_s**2<=44);
  }
});

test('rotating the full anisotropic drag tensor rotates the implicit physical response',()=>{
 const c=Math.SQRT1_2,unrotated=porousTensor(2,9,[0,1,0],1),rotated=porousTensor(2,9,[-c,c,0],1);
 const local=implicitTensorDragExchange(3,5,[c,-c,0],[0,0,0],unrotated,.1),world=implicitTensorDragExchange(3,5,[1,0,0],[0,0,0],rotated,.1);
 assert.ok(Math.abs(world[0]-(c*local[0]-c*local[1]))<1e-14);assert.ok(Math.abs(world[1]-(c*local[0]+c*local[1]))<1e-14);
 assert.ok(Math.abs(world[1])>.01,'oblique fibers must exchange transverse momentum');
 const fluid=[1-world[0]/3,-world[1]/3,-world[2]/3],solid=world.map(v=>v/5);
 assert.ok(3*fluid.reduce((s,v)=>s+v*v,0)+5*solid.reduce((s,v)=>s+v*v,0)<3);
});

test('competing absorption and implicit pore flow conserve both reservoirs', () => {
  const p=planPoreAbsorption([1,2],[.5,0],[1,2],[{cell:0,vertex:0,volume_m3:2},{cell:0,vertex:1,volume_m3:2},{cell:1,vertex:1,volume_m3:4}]);
  assert.ok(p.boundVolumeDelta_m3[0]!<=.5&&p.boundVolumeDelta_m3[1]!<=2);
  assert.ok(p.freeVolumeDelta_m3[0]!>=-1&&p.freeVolumeDelta_m3[1]!>=-2);
  assert.ok(Math.abs(p.freeVolumeDelta_m3.reduce((a,b)=>a+b,0)+p.boundVolumeDelta_m3.reduce((a,b)=>a+b,0))<1e-14);
  const r=diffusePoreWater([1,0,0],[1,1,1],[{a:0,b:1,conductance_m3_s:1},{a:1,b:2,conductance_m3_s:1}],100);
  assert.ok(r.volume_m3.every(v=>v>0&&v<1));assert.ok(Math.abs(r.volume_m3.reduce((a,b)=>a+b,0)-1)<1e-12);
});

test('MAC interpolation partitions structural masses and impulse publications apply once',()=>{
  const state=createClothState(prepareClothTopology(dragTriangle,[material]));
  const identity={generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0};
  const p=prepareUniformClothStencil({origin_m:[-.4,0,-.4],cellSize_m:[.05,.05,.05]}, {state,data:dragTriangle,materials:[material],identity,dt_s:.01});
  let splitMass=0;for(let i=0;i<p.coefficients.length;i+=4)splitMass+=p.coefficients[i]!;
  assert.ok(Math.abs(splitMass-3*state.dryMass_kg.reduce((a,b)=>a+b,0))<1e-8);
  const consumer=new ClothImpulseConsumer(),publication={...identity,mechanism:'drag' as const,impulse_Ns:new Float32Array(9).fill(.001)};
  consumer.apply(state,publication,1000);assert.throws(()=>consumer.apply(state,publication,1000),/once/);
});

test('local water exchange measures shared-corner volume and preserves untouched phi',()=>{
 const phi=new Float32Array([-.1,-.1,-.1,-.1,.1,.1,.1,.1]);
 assert.equal(tetrahedralWaterFill(Array.from(phi)),.5);
 const cells=[{vertices:[0,1,2,3,4,5,6,7],capacity_m3:1}];
 const capture=planLocalLevelSetExchange(phi,[1,1,1,1,0,0,0,0],cells,-.1,.1);
 assert.ok(capture.freeDelta_m3<=0&&capture.freeDelta_m3>=-.1);
 assert.ok(Math.abs(capture.freeDelta_m3+.1)<1e-7);
 assert.deepEqual(capture.phi_m.slice(4),phi.slice(4));
 const release=planLocalLevelSetExchange(capture.phi_m,[1,1,1,1,0,0,0,0],cells,.1,.1);
 assert.ok(Math.abs(release.freeDelta_m3-.1)<1e-7);
});

test('mixture pressure face is the positive Schur complement of both inertias',()=>{
 const s=[{fraction:.2,mass:400,velocity:2}],m=mixturePressureFace(1,1000,1,s);
 assert.ok(Math.abs(m.velocity-1.2)<1e-14);assert.equal(m.mobility,.9);
 const pressureImpulse=-30,fluidNext=1+pressureImpulse/1000,solidNext=2+.2*pressureImpulse/400;
 assert.ok(Math.abs(.8*fluidNext+.2*solidNext-(m.velocity+pressureImpulse/1000*m.mobility))<1e-14);
 assert.equal(mixturePressureFace(1,1000,1,[{...s[0]!,fixed:true}]).mobility,.8);
});

test('implicit manifold Darcy gravity drains downward with conservative water and momentum',()=>{
 const data:ClothTopologyData={restPositions_m:[0,1,0,0,0,0],initialVelocities_m_s:[2,0,0,0,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]};
 const m={...material,surfaceTension_N_m:0},state=createClothState(prepareClothTopology(data,[m]));
 state.waterVolume_m3[0]=.8*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
 const before=state.waterVolume_m3[0]!,momentum=state.velocities_m_s[0]!*(state.dryMass_kg[0]!+1000*before);
 const result=advanceClothPoreTransport({state,data,materials:[m],dt_s:100,gravity_m_s2:[0,-9.80665,0]});
 assert.ok(state.waterVolume_m3[1]!>0&&state.waterVolume_m3[0]!<before);
 assert.ok(Math.abs(state.waterVolume_m3[0]!+state.waterVolume_m3[1]!-before)<1e-18);
 let afterMomentum=0;for(let i=0;i<2;i++)afterMomentum+=(state.dryMass_kg[i]!+1000*state.waterVolume_m3[i]!)*state.velocities_m_s[3*i]!;
 assert.ok(Math.abs(momentum-afterMomentum)<1e-14);assert.ok(result.movedVolume_m3>0);
});

test('shell pore transport wets neighboring controls without moving dry geometry',()=>{
 const state=createClothState(prepareClothTopology(dragTriangle,[material]));state.waterVolume_m3[0]=.5*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
 const positions=state.positions_m.slice(),before=state.waterVolume_m3[0]!;
 advanceClothPoreTransport({state,data:dragTriangle,materials:[material],dt_s:.01,gravity_m_s2:[0,0,0]});
 assert.ok(state.waterVolume_m3[1]!>0&&state.waterVolume_m3[2]!>0);
 assert.ok(Math.abs(state.waterVolume_m3.reduce((a,b)=>a+b,0)-before)<1e-18);assert.deepEqual(state.positions_m,positions);
});


test('submerged pore capture changes capacity exactly once and preserves phi',()=>{
 const phi=new Float32Array(8).fill(-1),vertices=[0,1,2,3,4,5,6,7];
 const result=planLocalLevelSetExchange(phi,new Float32Array(8).fill(1),[{vertices,previousCapacity_m3:.9,capacity_m3:.7}],-.2,.1);
 assert.deepEqual(result.phi_m,phi);assert.ok(Math.abs(result.freeDelta_m3+.2)<1e-15);
 const motion=planLocalCapacityRemap(phi,new Float32Array(8).fill(1),[
  {vertices,previousCapacity_m3:.9,capacity_m3:.7},{vertices,previousCapacity_m3:.7,capacity_m3:.9}],.1);
 assert.deepEqual(motion.phi_m,phi);assert.equal(motion.freeDelta_m3,0);
});

test('moving material transfers native free-water momentum within its wet support',()=>{
 const sites=[{p:[0,0,0],oldVolume_m3:1,deltaVolume_m3:-.2,velocity_m_s:3},{p:[1,0,0],oldVolume_m3:1,deltaVolume_m3:.2,velocity_m_s:0}];
 const velocity=remapCapacityMomentum(sites);
 assert.equal(velocity[0],3);assert.ok(Math.abs(velocity[1]!-.5)<1e-15);
 assert.ok(Math.abs(sites.reduce((s,v,i)=>s+(v.oldVolume_m3+v.deltaVolume_m3)*velocity[i]!,0)-3)<1e-15);
 assert.throws(()=>remapCapacityMomentum([sites[0]!,{...sites[1]!,p:[3,0,0]}]),/disconnected/);
});

test('manifold water is stationary under projected hydrostatic pressure and uniform free fall',()=>{
 const data:ClothTopologyData={restPositions_m:[0,1,0,0,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},m={...material,surfaceTension_N_m:0};
 for(const mode of ['hydrostatic','freefall']){
  const state=createClothState(prepareClothTopology(data,[m]));state.waterVolume_m3[0]=.8*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
  const before=state.waterVolume_m3.slice(),acceleration=new Float64Array([0,mode==='hydrostatic'?9.80665:-9.80665,0,0,mode==='hydrostatic'?9.80665:-9.80665,0]);
  const result=advanceClothPoreTransport({state,data,materials:[m],dt_s:100,gravity_m_s2:[0,-9.80665,0],
   ...(mode==='hydrostatic'?{pressureAcceleration_m_s2:acceleration}:{solidAcceleration_m_s2:acceleration})});
  assert.deepEqual(state.waterVolume_m3,before,mode);assert.equal(result.movedVolume_m3,0);
 }
});


test('pressure MAC phase fractions equal the average of the same two native cell capacities',()=>{
 const state=createClothState(prepareClothTopology(dragTriangle,[material])),fields={origin_m:[-.4,0,-.4] as const,cellSize_m:[.05,.05,.05] as const};
 state.waterVolume_m3[0]=.2*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
 state.positions_m[0]!+=.007;
 const step={state,data:dragTriangle,materials:[material],identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001};
 const table=prepareClothPorousCapacity(fields,step),byCell=new Map(table.cells.map(c=>[c.p.join(','),c.solidFraction+c.boundFraction])),stencil=prepareUniformClothStencil(fields,step),V=fields.cellSize_m.reduce((a,b)=>a*b,1);
 stencil.coordinates.forEach((p,i)=>{let first=stencil.anchors[8*i+3]!;for(let a=0;a<3;a++){
  const count=stencil.anchors[8*i+4+a]!;let actual=0;for(let j=0;j<count;j++)actual+=stencil.occupiedVolumes_m3[first+j]!/V;first+=count;
  if(!count)continue;const q=[...p];q[a]!++;const expected=.5*((byCell.get(p.join(','))??0)+(byCell.get(q.join(','))??0));assert.ok(Math.abs(actual-expected)<1e-8);
 }});
});


test('hard-clamp apertures conserve fiber and water with the same adjoint MAC basis',()=>{
 const state=createClothState(prepareClothTopology(dragTriangle,[material])),fields={origin_m:[-.4,0,-.4] as const,cellSize_m:[.05,.05,.05] as const};
 state.positions_m[1]!+=.013;state.waterVolume_m3[0]=.2*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
 const apertures=new Map<string,number>();for(let v=0;v<state.waterVolume_m3.length;v++)for(const cell of clothCellWeights(fields,state.positions_m.subarray(3*v,3*v+3),true))apertures.set(cell.p.join(','),cell.p[1]<4?.15:1);
 setClothHardApertures(fields,apertures);
 const table=prepareClothPorousCapacity(fields,{state}),V=fields.cellSize_m.reduce((a,b)=>a*b,1);
 const solid=table.cells.reduce((sum,c)=>sum+c.solidFraction*apertures.get(c.p.join(','))!*V,0),bound=table.cells.reduce((sum,c)=>sum+c.boundFraction*apertures.get(c.p.join(','))!*V,0);
 assert.ok(Math.abs(solid-state.solidVolume_m3.reduce((a,b)=>a+b,0))<1e-18);assert.ok(Math.abs(bound-state.waterVolume_m3[0]!)<1e-18);
 const stencil=prepareUniformClothStencil(fields,{state,data:dragTriangle,materials:[material],identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001});
 assert.ok(Math.abs(stencil.occupiedVolumes_m3.reduce((a,b)=>a+b,0)-3*(solid+bound))<1e-11);
 const first=state.positions_m.subarray(0,3);setClothHardApertures(fields,new Map([...apertures.keys()].map(k=>[k,0])));assert.throws(()=>clothCellWeights(fields,first),/no open local/);
});


test('bound transport through a submerged manifold corridor preserves the native free surface',()=>{
 const index=(x:number,y:number,z:number)=>x+5*(y+2*z),phi=new Float32Array(20).fill(-.12);
 const cells=Array.from({length:4},(_,x)=>({vertices:Array.from({length:8},(_,c)=>index(x+(c&1),(c>>1)&1,(c>>2)&1)),previousCapacity_m3:x===0?.8:1,capacity_m3:x===3?.8:1}));
 const result=planLocalCapacityRemap(phi,new Float32Array(phi.length).fill(1),cells,.01);
 assert.deepEqual(result.phi_m,phi);assert.equal(result.freeDelta_m3,0);
 const velocity=remapCapacityMomentum(cells.map((c,i)=>({p:[i,0,0],oldVolume_m3:c.previousCapacity_m3,deltaVolume_m3:result.cellDelta_m3[i]!,velocity_m_s:1})));
 assert.ok(velocity.every(v=>Math.abs(v-1)<1e-15));
});


test('source drag and suction use distinct canonical axes and capture uses grid diffusion time',()=>{
 const cloth=porousDragTensor(2,9,[0,0,1],1),rod=porousDragTensor(2,9,[0,0,1],0);
 assert.deepEqual([cloth[0],cloth[4],cloth[8]],[9,9,2]);assert.deepEqual([rod[0],rod[4],rod[8]],[2,2,9]);
 const force=17,speed=quasiStaticPorousSpeed(force,3,8,1.6);assert.ok(speed<force/3);assert.ok(Math.abs((3+8*speed**1.6)*speed-force)<1e-12);
 const fraction=porousCaptureFraction(.4,material,.02,1e-8,.7);assert.ok(fraction>0&&fraction<.1);
 assert.ok(Math.abs(porousCaptureFraction(.4,material,.04,1e-8,.7)*4-fraction)<1e-15);
 assert.equal(porousCaptureFraction(.4,material,.02,1e-8,0),0);
});

test('Richards manifold flux has quadratic low-saturation scaling',()=>{
 const data:ClothTopologyData={restPositions_m:[0,0,0,.1,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]};
 const moved=[.2,.4].map(saturation=>{const state=createClothState(prepareClothTopology(data,[material]));state.waterVolume_m3[0]=saturation*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
  return advanceClothPoreTransport({state,data,materials:[material],dt_s:1e-6,gravity_m_s2:[0,0,0],physics:{nonlinearDrag:false}}).movedVolume_m3;
 });assert.ok(Math.abs(moved[1]!/moved[0]!-4)<1e-5);
});


test('grid drag coefficients use deposited packing and partition the mixture exactly once',()=>{
 const state=createClothState(prepareClothTopology(dragTriangle,[material])),fields={origin_m:[-.4,0,-.4] as const,cellSize_m:[.05,.05,.05] as const};
 const step={state,data:dragTriangle,materials:[material],identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001},stencil=prepareUniformClothStencil(fields,step),cells=prepareGridDragMaterial(fields,stencil,step),V=fields.cellSize_m.reduce((a,b)=>a*b,1);
 for(const cell of cells){
  const psi=cell.fiberVolume_m3/V,c=porousDragCoefficients(psi,0,material,false);
  assert.notEqual(psi,material.solidFraction,'subresolution fibers have a diluted grid packing');
  assert.ok(Math.abs(cell.linear[4]!-c.alongFiber_kg_m3_s)<1e-8*c.alongFiber_kg_m3_s);
  assert.ok(Math.abs(cell.linear[0]!-c.acrossFiber_kg_m3_s)<1e-8*c.acrossFiber_kg_m3_s);
  assert.ok(Math.abs(cell.members.reduce((sum,m)=>sum+m.share,0)-1)<1e-14);
 }
});

test('imported manifold substeps preserve the outer nonlinear force predictor interval',()=>{
 const data:ClothTopologyData={restPositions_m:[0,0,0,.1,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},prepared=prepareClothTopology(data,[material]);
 const a=createClothState(prepared),b=createClothState(prepared);a.waterVolume_m3[0]=b.waterVolume_m3[0]=.6*(a.controlVolume_m3[0]!-a.solidVolume_m3[0]!);
 advanceClothPoreTransport({state:a,data,materials:[material],dt_s:.01,gravity_m_s2:[0,0,0],physics:{manifoldSubsteps:4}});
 for(let i=0;i<4;i++)advanceClothPoreTransport({state:b,data,materials:[material],dt_s:.0025,manifoldPredictorDt_s:.01,gravity_m_s2:[0,0,0]});
 assert.deepEqual(a.waterVolume_m3,b.waterVolume_m3);assert.deepEqual(a.velocities_m_s,b.velocities_m_s);
});

test('shell manifold uses the source finite-element metric rather than diagonal-edge diffusion',()=>{
 const data:ClothTopologyData={restPositions_m:[0,0,0,.1,0,0,0,0,.1,.1,0,.1],triangles:[0,1,3,0,3,2],rodOffsets:[0],rodIndices:[]};
 const state=createClothState(prepareClothTopology(data,[material]));state.waterVolume_m3[0]=.5*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
 advanceClothPoreTransport({state,data,materials:[material],dt_s:1e-5,gravity_m_s2:[0,0,0],physics:{nonlinearDrag:false}});
 assert.ok(state.waterVolume_m3[1]!>0&&state.waterVolume_m3[2]!>0);
 assert.ok(Math.abs(state.waterVolume_m3[1]!-state.waterVolume_m3[2]!)<1e-18,'orthogonal directions have equal metric');
 assert.ok(state.waterVolume_m3[3]!<state.waterVolume_m3[1]!*1e-4,'right-triangle diagonal has zero direct FEM conductance');
});

test('fixed source control volume uses deformed gradients on stretched rods',()=>{
 const data:ClothTopologyData={restPositions_m:[0,0,0,.1,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]};
 const values=[1,2].map(stretch=>{const state=createClothState(prepareClothTopology(data,[material]));state.positions_m[3]!*=stretch;state.waterVolume_m3[0]=.5*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
  return advanceClothPoreTransport({state,data,materials:[material],dt_s:1e-6,gravity_m_s2:[0,0,0],physics:{nonlinearDrag:false,varyingSolidFraction:false}}).movedVolume_m3;
 });assert.ok(Math.abs(values[0]!/values[1]!-4)<1e-5,'fixed Gauss volume multiplies two current spatial gradients');
});

test('nonmonotone manifold flux preserves both ends while enforcing donor positivity',()=>{
 const valid=[{a:0,b:1,volume_m3:.1},{a:1,b:2,volume_m3:.05}],old=new Float64Array([.2,0,0]);
 assert.deepEqual(acceptPositiveManifoldFlux(old,valid),{water:new Float64Array([.1,.05,.05]),fluxes:valid,limited:false});
 const limited=acceptPositiveManifoldFlux(old,[{a:0,b:1,volume_m3:.4},{a:2,b:1,volume_m3:.1}]);
 assert.equal(limited.limited,true);assert.ok(limited.water.every(v=>v>=0));assert.equal(limited.water.reduce((a,b)=>a+b,0),.2);
 assert.deepEqual(limited.fluxes.map(f=>f.volume_m3),[.2,0]);
});

test('obtuse shell retains its actual metric and conservatively limits wet-front undershoot',()=>{
 const data:ClothTopologyData={restPositions_m:[0,0,0,.1,0,0,.02,0,.01],triangles:[0,1,2],rodOffsets:[0],rodIndices:[]};
 const state=createClothState(prepareClothTopology(data,[material]));state.waterVolume_m3[0]=.5*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);
 const before=state.waterVolume_m3.reduce((a,b)=>a+b,0);for(let i=0;i<3;i++)state.velocities_m_s[3*i]=2;
 const result=advanceClothPoreTransport({state,data,materials:[material],dt_s:.01,gravity_m_s2:[0,0,0],physics:{nonlinearDrag:false}});
 assert.ok(result.limitedVolume_m3>0);assert.ok(state.waterVolume_m3.every(v=>v>=0));assert.ok(Math.abs(state.waterVolume_m3.reduce((a,b)=>a+b,0)-before)<1e-18);
 assert.ok(state.velocities_m_s.filter((_,i)=>i%3===0).every(v=>Math.abs(v-2)<1e-10),'accepted water and momentum must use identical paired fluxes');
});

test('nonuniform packing retains material pore radius and equilibrates actual pore pressure',()=>{
 const low=porousSuctionPressure(.2,material).alongFiber_Pa,high=porousSuctionPressure(.6,material).alongFiber_Pa;
 assert.ok(Math.abs(high/low-6)<1e-12,'pore radius is fixed by rest material, not current packing');
 const data:ClothTopologyData={restPositions_m:[0,0,0,.1,0,0],solidFraction:[.2,.6],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},prepared=prepareClothTopology(data,[material]);
 const equilibrium=createClothState(prepared),saturations=[.2,1-low*.8/high];
 for(let i=0;i<2;i++)equilibrium.waterVolume_m3[i]=saturations[i]!*(equilibrium.controlVolume_m3[i]!-equilibrium.solidVolume_m3[i]!);
 const before=equilibrium.waterVolume_m3.slice();advanceClothPoreTransport({state:equilibrium,data,materials:[material],dt_s:1,gravity_m_s2:[0,0,0],physics:{nonlinearDrag:false}});
 for(let i=0;i<2;i++)assert.ok(Math.abs(equilibrium.waterVolume_m3[i]!-before[i]!)<1e-14*before[i]!,`equal pore potential must not flow at node ${i}`);
 const unequal=createClothState(prepared);for(let i=0;i<2;i++)unequal.waterVolume_m3[i]=.2*(unequal.controlVolume_m3[i]!-unequal.solidVolume_m3[i]!);
 const receiver=unequal.waterVolume_m3[1]!;advanceClothPoreTransport({state:unequal,data,materials:[material],dt_s:.01,gravity_m_s2:[0,0,0],physics:{nonlinearDrag:false}});
 assert.ok(unequal.waterVolume_m3[1]!>receiver,'equal saturation still flows toward higher suction material');
});

test('dry fabric corridors cannot transfer a capacity correction between disconnected pools',()=>{
 const nx=8,index=(x:number,y:number,z:number)=>x+nx*(y+2*z),line=[-1,-.2,.3,1,1,.3,-.2,-1];
 const phi=Float32Array.from({length:4*nx},(_,i)=>line[i%nx]!);
 const cells=Array.from({length:7},(_,x)=>({vertices:Array.from({length:8},(_,c)=>index(x+(c&1),(c>>1)&1,(c>>2)&1)),previousCapacity_m3:1,capacity_m3:x===0?.8:1}));
 const result=planLocalCapacityRemap(phi,new Float32Array(phi.length).fill(1),cells,1);
 assert.ok(result.phi_m.some((p,i)=>p!==phi[i]),'left pool must locally represent the displaced water');
 for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=5;x<8;x++)assert.equal(result.phi_m[index(x,y,z)],phi[index(x,y,z)],'disconnected right surface remains bit-identical');
 assert.ok(Math.abs(result.cellDelta_m3.slice(0,3).reduce((a,b)=>a+b,0))<1e-6);assert.equal(result.cellDelta_m3.slice(4).reduce((a,b)=>a+b,0),0);
});

test('finite source attachment flags do not eliminate fluid-coupled or transported momentum',()=>{
 const data={...dragTriangle,fixed:[1,1,1]},state=createClothState(prepareClothTopology(data,[material])),fields={origin_m:[-.4,0,-.4] as const,cellSize_m:[.05,.05,.05] as const};
 const step={state,data,materials:[material],identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001};
 const hard=prepareUniformClothStencil(fields,step),finite=prepareUniformClothStencil(fields,{...step,kinematicFixed:new Uint8Array(3)});
 assert.ok(hard.coefficients.filter((_,i)=>i%4===0).every(m=>m===0));assert.ok(finite.coefficients.filter((_,i)=>i%4===0).every(m=>m>0));
 const rod:ClothTopologyData={restPositions_m:[0,1,0,0,0,0],fixed:[0,1],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},m={...material,surfaceTension_N_m:0},prepared=prepareClothTopology(rod,[m]);
 const velocities=[undefined,new Uint8Array(2)].map(kinematicFixed=>{
  const s=createClothState(prepared);s.waterVolume_m3[0]=.8*(s.controlVolume_m3[0]!-s.solidVolume_m3[0]!);s.velocities_m_s[0]=2;
  advanceClothPoreTransport({state:s,data:rod,materials:[m],dt_s:1,gravity_m_s2:[0,-9.80665,0],kinematicFixed});return s.velocities_m_s[3]!;
 });assert.equal(velocities[0],0);assert.ok(velocities[1]!>0,'a finite attachment receives incoming fluid momentum');
});

test('fully packed source controls retain zero pore capacity and finite deposited-grid drag',()=>{
 const data={...dragTriangle,solidFraction:[1,1,1]},state=createClothState(prepareClothTopology(data,[material])),fields={origin_m:[-.4,0,-.4] as const,cellSize_m:[.05,.05,.05] as const};
 const step={state,data,materials:[material],identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001};
 const stencil=prepareUniformClothStencil(fields,step),grid=prepareGridDragMaterial(fields,stencil,step),capacity=prepareClothPorousCapacity(fields,step);
 assert.ok(stencil.coordinates.length>0&&grid.length>0);assert.ok(state.solidFraction.every(v=>v===1));
 for(let i=0;i<state.controlVolume_m3.length;i++)assert.equal(state.controlVolume_m3[i]!-state.solidVolume_m3[i]!,0);
 assert.ok(grid.every(c=>c.fiberVolume_m3<capacity.cellVolume_m3&&c.linear.every(Number.isFinite)&&c.linear[0]!>0));
 assert.ok(Math.abs(capacity.solidVolume_m3-state.controlVolume_m3.reduce((a,b)=>a+b,0))<1e-18);
 assert.equal(advanceClothPoreTransport({state,data,materials:[material],dt_s:1,gravity_m_s2:[0,-9.8,0]}).movedVolume_m3,0);
});

test('quantized interface capture closes free and bound volume without a receiver fixed point',()=>{
 let phi:Float32Array=new Float32Array([-.001,.001,-.001,.001,-.001,.001,-.001,.001]),capacity=2.3887872e-8,bound=0;
 const initial=tetrahedralWaterFill(Array.from(phi))*capacity;
 for(let step=0;step<100;step++){
  const requests=[1e-11,1.7e-11],result=planLocalPoreExchange(phi,[{vertices:[0,1,2,3,4,5,6,7],capacity_m3:capacity}],requests.map(requestedBound_m3=>({requestedBound_m3,capacityWeights:[{cell:0,weight:1}],phiWeights:Array.from({length:8},(_,vertex)=>({vertex,weight:1/8}))})),.00288);
  const accepted=result.boundByReceiver_m3.reduce((a,b)=>a+b,0);assert.ok(accepted>0&&accepted<=requests.reduce((a,b)=>a+b,0));
  assert.ok(Math.abs(result.plan.freeDelta_m3+accepted)<1e-23);
  for(let v=0;v<2;v++){const share=result.receiverShares[0]!.find(s=>s.v===v)!.share;assert.ok(Math.abs(-result.plan.cellDelta_m3[0]!*share-result.boundByReceiver_m3[v]!)<1e-23);}
  phi=result.plan.phi_m;capacity-=accepted;bound+=accepted;
  assert.ok(Math.abs(tetrahedralWaterFill(Array.from(phi))*capacity+bound-initial)<1e-22);
 }
});

test('local pore capture keeps disconnected submerged and interface receivers separate',()=>{
 const phi=new Float32Array([-.001,.001,-.001,.001,-.001,.001,-.001,.001,...Array(8).fill(-.1)]),cells=[0,8].map(base=>({vertices:Array.from({length:8},(_,i)=>base+i),capacity_m3:1e-8}));
 const receivers=[0,1].map(cell=>({requestedBound_m3:1e-10,capacityWeights:[{cell,weight:1}],phiWeights:Array.from({length:8},(_,i)=>({vertex:8*cell+i,weight:1/8}))}));
 const result=planLocalPoreExchange(phi,cells,receivers,.002);
 assert.deepEqual(result.plan.phi_m.slice(8),phi.slice(8));assert.equal(result.boundByReceiver_m3[1],1e-10);
 for(let cell=0;cell<2;cell++)assert.ok(Math.abs(result.plan.cellDelta_m3[cell]!+result.boundByReceiver_m3[cell]!)<1e-22);
 assert.deepEqual(result.receiverShares.map(row=>row.map(s=>s.v)),[[0],[1]]);
});

test('local pore release closes the same quantized surface and momentum margins',()=>{
 let phi:Float32Array=new Float32Array([-.001,.001,-.001,.001,-.001,.001,-.001,.001]),capacity=1.9e-8,bound=1e-9;
 const initial=tetrahedralWaterFill(Array.from(phi))*capacity+bound;
 for(let step=0;step<10;step++){
  const result=planLocalPoreExchange(phi,[{vertices:[0,1,2,3,4,5,6,7],capacity_m3:capacity}],[{requestedBound_m3:-1e-10,capacityWeights:[{cell:0,weight:1}],phiWeights:Array.from({length:8},(_,vertex)=>({vertex,weight:1/8}))}],.00288);
  const transferred=result.boundByReceiver_m3[0]!;assert.ok(transferred<0&&transferred>= -1e-10);assert.equal(result.receiverShares[0]![0]!.share,1);
  phi=result.plan.phi_m;capacity-=transferred;bound+=transferred;
  assert.ok(Math.abs(tetrahedralWaterFill(Array.from(phi))*capacity+bound-initial)<1e-22);
 }
});

test('fluid stencils use the nonfluid force predictor without integrating accepted cloth twice',()=>{
 const state=createClothState(prepareClothTopology(dragTriangle,[material])),before=state.velocities_m_s.slice(),predictor=Float64Array.from(before,(v,i)=>v+(i%3===1?-.00980665:0));
 const stencil=prepareUniformClothStencil({origin_m:[-.4,0,-.4],cellSize_m:[.05,.05,.05]},{state,data:dragTriangle,materials:[material],solidPredictorVelocity_m_s:predictor,identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001});
 for(let vertex=0;vertex<state.dryMass_kg.length;vertex++)for(let a=0;a<3;a++){
  const first=stencil.ranges[6*vertex+2*a]!,count=stencil.ranges[6*vertex+2*a+1]!;for(let j=0;j<count;j++)assert.ok(Math.abs(stencil.coefficients[4*stencil.gather[first+j]!+3]!-predictor[3*vertex+a]!)<1e-8);
 }assert.deepEqual(state.velocities_m_s,before);
});

 test('disabled solid solve remains an immovable fluid support with finite source attachments',()=>{
 const state=createClothState(prepareClothTopology(dragTriangle,[material])),fields={origin_m:[-.4,0,-.4] as const,cellSize_m:[.05,.05,.05] as const};
 const step={state,data:dragTriangle,materials:[material],kinematicFixed:new Uint8Array(state.dryMass_kg.length),physics:{solveSolid:false},identity:{generation:state.generation,topologyRevision:state.topologyRevision,tick:0,time_s:0},dt_s:.001};
 const stencil=prepareUniformClothStencil(fields,step);assert.ok(stencil.coefficients.filter((_,i)=>i%4===0).every(m=>m===0));
 });

test('source nonlinear manifold drag uses the full rotated Gauss force and outer dt',()=>{
 const expected=.6*.02*5;assert.ok(Math.abs(manifoldForcePredictorSpeed(.4,.02,[3,4,0])-expected)<1e-15);
 assert.ok(Math.abs(manifoldForcePredictorSpeed(.4,.02,[0,0,5])-expected)<1e-15);
 assert.equal(manifoldForcePredictorSpeed(.4,.02,[0,0,0]),0);
 assert.equal(manifoldForcePredictorSpeed(.4,.01,[3,4,0]),expected/2);
});

test('interface capture routes donor momentum through actual face-connected local liquid',()=>{
 const point=(x:number,y:number,z:number)=>x+6*(y+2*z),height=[0.26280996203422546, 0.3823436200618744, 0.5038178563117981, 0.6843949556350708, 0.0954892635345459, 0.38256651163101196];
 const phi=Float32Array.from({length:24},(_,i)=>Math.floor(i/6)%2-height[i%6]!);
 const cells=Array.from({length:5},(_,x)=>({vertices:Array.from({length:8},(_,c)=>point(x+(c&1),(c>>1)&1,c>>2)),capacity_m3:1}));
 const requested=[0.07747629624791444, 0.05561885705683381, 0.0016493235714733602, 0.06392460397910327],receivers=requested.map((requestedBound_m3,v)=>({requestedBound_m3,capacityWeights:[{cell:v,weight:.5},{cell:v+1,weight:.5}],phiWeights:[{vertex:point(v+1,0,0),weight:.5},{vertex:point(v+1,1,0),weight:.5}]}));
 const result=planLocalPoreExchange(phi,cells,receivers,1),credited=new Float64Array(receivers.length);let transit=false;
 result.receiverShares.forEach((row,cell)=>{assert.ok(Math.abs(row.reduce((sum,s)=>sum+s.share,0)-1)<1e-12);for(const share of row){credited[share.v]!-=result.plan.cellDelta_m3[cell]!*share.share;if(cell!==share.v&&cell!==share.v+1)transit=true;}});
 assert.ok(transit,'fixture requires a wet-cell transit path, not direct receiver adjacency');
 credited.forEach((volume,v)=>assert.ok(Math.abs(volume-result.boundByReceiver_m3[v]!)<1e-12));
 assert.ok(Math.abs(result.plan.freeDelta_m3+result.boundByReceiver_m3.reduce((a,b)=>a+b,0))<1e-12);
});

test('source Gauss nonlinear mobility retains force normal to a stationary shell',()=>{
 const data:ClothTopologyData={restPositions_m:[0,0,0,.1,0,0,0,0,.1],triangles:[0,1,2],rodOffsets:[0],rodIndices:[]},prepared=prepareClothTopology(data,[material]);
 const run=(gravity:number,nonlinearDrag:boolean)=>{const state=createClothState(prepared);state.waterVolume_m3[0]=.5*(state.controlVolume_m3[0]!-state.solidVolume_m3[0]!);return advanceClothPoreTransport({state,data,materials:[material],dt_s:.1,gravity_m_s2:[0,gravity,0],physics:{solveSolid:false,nonlinearDrag}}).movedVolume_m3;};
 assert.ok(run(-9.80665,true)<run(0,true),'the full source force predictor changes nonlinear C even when its extra component is normal to the manifold');
 assert.equal(run(-9.80665,false),run(0,false),'normal force does not create tangential Darcy drive');
});


test('capture surface support includes every actual staggered MAC donor',()=>{
 const cells=clothMacCaptureReservoir({origin_m:[0,0,0],cellSize_m:[1,1,1]},[1.5,1.5,1.5]);
 assert.equal(cells.reduce((sum,c)=>sum+c.weight,0),3);assert.equal(cells.find(c=>c.p.join(',')==='1,1,1')!.weight,1.5);
 assert.equal(cells.find(c=>c.p.join(',')==='1,0,1')!.weight,.25,'a real lower MAC donor extends beyond the nodal interpolation cells');
 const corners=new Set(cells.flatMap(c=>Array.from({length:8},(_,i)=>c.p.map((v,a)=>v+((i>>a)&1)).join(','))));assert.ok(corners.has('1,0,1'));
});
