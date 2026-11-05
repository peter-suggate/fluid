import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {prepareClothTopology,createClothState} from '../lib/cloth/topology';
import {membraneCoordinateDerivatives,stepCloth} from '../lib/cloth/dynamics';
import type {ClothMaterial} from '../lib/cloth/types';

test('explicit membrane relaxation controls its full force and Hessian independently of bending',()=>{
 const data={restPositions_m:[0,0,0,1,0,0,0,1,0],triangles:[0,1,2],rodOffsets:[0],rodIndices:[]},base={...DEFAULT_CLOTH_MATERIAL,mechanicalDamping_s:.07,membraneDamping_s:0,bendingDamping_s:.5};
 const a=prepareClothTopology(data,[base]),b=prepareClothTopology(data,[{...base,membraneDamping_s:.003}]),state=createClothState(a),dt=.001;state.positions_m[3]=1.1;
 const dry=membraneCoordinateDerivatives(a,state,0,1,0,data.restPositions_m,dt),wet=membraneCoordinateDerivatives(b,state,0,1,0,data.restPositions_m,dt);
 assert.ok(Math.abs(wet.gradient/dry.gradient-4)<1e-12);assert.ok(Math.abs(wet.diagonal/dry.diagonal-4)<1e-12);
});
test('rod stretch viscosity can be disabled while bending and twist dissipation stay enabled',()=>{
 const data={restPositions_m:[0,0,0,.01,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1],fixed:[1,0],initialVelocities_m_s:[0,0,0,.1,0,0]};
 const run=(material:ClothMaterial)=>{const p=prepareClothTopology(data,[material]),s=createClothState(p);const report=stepCloth(p,s,.001,{gravity_m_s2:[0,0,0],iterations:128,tolerance_m:1e-13});assert.ok(report.converged);return s.positions_m[3]!-.01;};
 const base={...DEFAULT_CLOTH_MATERIAL,mechanicalDamping_s:.05,rodStretchDamping_s:0,rodBendingDamping_s:.1,rodTwistDamping_s:.2},undamped=run({...base,mechanicalDamping_s:0}),bendingOnly=run(base),allModes=run({...base,rodStretchDamping_s:.003});
 assert.ok(Math.abs(undamped-bendingOnly)<1e-14);assert.ok(allModes>0&&allModes<bendingOnly);
});
