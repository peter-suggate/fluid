import test from 'node:test';import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {createClothState,prepareClothTopology} from '../lib/cloth/topology';
import {cohesionPairValue,prepareClothCohesion,resolveClothCohesion,cohesionSolidReactions,type CohesionSolidSample} from '../lib/cloth/cohesion';
import {stepCloth} from '../lib/cloth/dynamics';
const fixture=()=>{const topology=prepareClothTopology({restPositions_m:[0,0,0,.01,0,0,0,0,.01,0,.02,0,.01,.02,0,0,.02,.01],triangles:[0,2,1,3,5,4],rodOffsets:[0],rodIndices:[]},[DEFAULT_CLOTH_MATERIAL]),state=createClothState(topology);for(let i=0;i<state.waterVolume_m3.length;i++)state.waterVolume_m3[i]=state.controlVolume_m3[i]!*(1-state.solidFraction[i]!)*.5;return{topology,state};};
const options={cellSize_m:.01,bucketSize_m:.04,bucketOrigin_m:[-.12,-.12,-.12] as const,freeParticleCount:1};
test('source cohesion requires actual liquid bridge occupancy and enabled free phase',()=>{
 const {topology,state}=fixture(),prepared=prepareClothCohesion(topology,state,options);assert.equal(prepared.candidates.length,2);assert.equal(prepared.samplePositions_m.length,18);
 assert.deepEqual(resolveClothCohesion(prepared,new Float64Array(6).fill(.01)),[]);
 const pairs=resolveClothCohesion(prepared,new Float64Array(6).fill(-.01));assert.equal(pairs.length,2);assert.ok(pairs.every(p=>p.force_N>0));assert.ok(pairs.every(p=>Math.abs(p.weights.reduce((a,b)=>a+b,0))<1e-14));
 assert.equal(prepareClothCohesion(topology,state,{...options,freeParticleCount:0}).samplePositions_m.length,0);assert.equal(prepareClothCohesion(topology,state,{...options,enabled:false}).samplePositions_m.length,0);
 assert.throws(()=>resolveClothCohesion(prepared,[NaN,0,0,0,0,0]),/native liquid phi/);assert.throws(()=>resolveClothCohesion(prepared,[0]),/sample count/);
 state.waterVolume_m3.fill(0);assert.equal(prepareClothCohesion(topology,state,options).candidates.length,0);
});

test('native-phi-gated rigid cohesion has fixed endpoint energy and paired linear/angular impulse',()=>{
 const {topology,state}=fixture(),sample:CohesionSolidSample={id:7,position_m:[.0025,.02,.0025],normal:[0,1,0],volume_m3:4*Math.PI*.005**3/3,bodyId:'clamp',bodyCenter_m:[.05,.02,.0025]},prepared=prepareClothCohesion(topology,state,{...options,softCohesion:false,solidSamples:[sample]});
 assert.equal(prepared.candidates.length,1);assert.equal(resolveClothCohesion(prepared,new Float64Array(3).fill(.01)).length,0);
 const pairs=resolveClothCohesion(prepared,new Float64Array(3).fill(-.01)),pair=pairs[0]!;assert.equal(pairs.length,1);assert.deepEqual(pair.targetPosition_m,sample.position_m);assert.equal(pair.weights.reduce((a,b)=>a+b,0),1);
 const h=1e-6,old=state.positions_m[1]!,d=cohesionPairValue(pair,state.positions_m,0,1);state.positions_m[1]=old+h;const plus=cohesionPairValue(pair,state.positions_m).energy_J;state.positions_m[1]=old-h;const minus=cohesionPairValue(pair,state.positions_m).energy_J;state.positions_m[1]=old;
 assert.ok(Math.abs(d.gradient_N-(plus-minus)/(2*h))<1e-12);assert.ok(Math.abs(d.diagonal_N_m-(plus+minus-2*d.energy_J)/(h*h))<1e-9);
 const report=stepCloth(topology,state,.001,{gravity_m_s2:[0,0,0],cohesionPairs:pairs,iterations:128,tolerance_m:1e-12}),reaction=report.contacts.reactions[0]!;assert.ok(report.converged);assert.equal(reaction.bodyId,'clamp');assert.equal(report.cohesionReactions![0]!.sampleId,7);
 const momentum=[0,0,0],angular=[0,0,0];for(let i=0;i<6;i++){const m=state.dryMass_kg[i]!+1000*state.waterVolume_m3[i]!,x=state.positions_m.slice(3*i,3*i+3),p=Array.from(state.velocities_m_s.slice(3*i,3*i+3),v=>m*v);for(let a=0;a<3;a++){momentum[a]!+=p[a]!;angular[a]!+=x[(a+1)%3]!*p[(a+2)%3]!-x[(a+2)%3]!*p[(a+1)%3]!;}}
 for(let a=0;a<3;a++){assert.ok(Math.abs(momentum[a]!+reaction.linearImpulse_N_s[a]!)<1e-12);const moment=sample.bodyCenter_m![(a+1)%3]!*reaction.linearImpulse_N_s[(a+2)%3]!-sample.bodyCenter_m![(a+2)%3]!*reaction.linearImpulse_N_s[(a+1)%3]!;assert.ok(Math.abs(angular[a]!+reaction.angularImpulse_N_m_s[a]!+moment)<1e-12);}
 const fixed={...pair,bodyId:undefined,bodyCenter_m:undefined};assert.equal(cohesionSolidReactions([fixed],state.positions_m,.001)[0]!.bodyId,undefined);
 assert.equal(prepareClothCohesion(topology,state,{...options,softCohesion:false,solidCohesion:false,solidSamples:[sample]}).candidates.length,0);
 const wrongFacing={...sample,normal:[0,-1,0] as const};assert.equal(prepareClothCohesion(topology,state,{...options,softCohesion:false,solidSamples:[wrongFacing]}).candidates.length,0);
});
test('source cohesion uses rest angle Gauss weights and reciprocal-pair halving',()=>{
 const {topology,state}=fixture(),prepared=prepareClothCohesion(topology,state,options),pair=resolveClothCohesion(prepared,new Float64Array(6).fill(-.01))[0]!;
 assert.deepEqual(pair.weights.slice(0,3),[.5,.25,.25]);const candidate=prepared.candidates[0]!;assert.ok(Math.abs(pair.force_N-.5*candidate.coefficient_N*candidate.orientation)<1e-15);
 // The source requires a bridge near its half-filled occupancy interval,
 // rather than pulling every saturated pair even when no meniscus exists.
 const thin=fixture();for(const v of[3,4,5])thin.state.positions_m[3*v+1]=.002;const q=prepareClothCohesion(thin.topology,thin.state,options);assert.equal(resolveClothCohesion(q,new Float64Array(q.samplePositions_m.length/3).fill(-.01)).length,0);
});
test('cohesion derivatives are energy-consistent and implicit forces preserve pair momentum',()=>{
 const {topology,state}=fixture(),prepared=prepareClothCohesion(topology,state,options),pairs=resolveClothCohesion(prepared,new Float64Array(6).fill(-.01)),pair=pairs[0]!,k=1,h=1e-6,old=state.positions_m[k]!,d=cohesionPairValue(pair,state.positions_m,0,1),energy=d.energy_J;
 state.positions_m[k]=old+h;const plus=cohesionPairValue(pair,state.positions_m).energy_J;state.positions_m[k]=old-h;const minus=cohesionPairValue(pair,state.positions_m).energy_J;state.positions_m[k]=old;
 assert.ok(Math.abs(d.gradient_N-(plus-minus)/(2*h))<1e-12);assert.ok(Math.abs(d.diagonal_N_m-(plus+minus-2*energy)/(h*h))<1e-9);
 const beforeGap=state.positions_m[10]!-state.positions_m[1]!,report=stepCloth(topology,state,.001,{gravity_m_s2:[0,0,0],cohesionPairs:pairs,iterations:128,tolerance_m:1e-10});assert.ok(report.converged);assert.ok(state.positions_m[10]!-state.positions_m[1]!<beforeGap);assert.ok(report.cohesionEnergy_J!>0);
 for(let a=0;a<3;a++){let momentum=0;for(let i=0;i<6;i++)momentum+=(state.dryMass_kg[i]!+1000*state.waterVolume_m3[i]!)*state.velocities_m_s[3*i+a]!;assert.ok(Math.abs(momentum)<1e-12);}
});
