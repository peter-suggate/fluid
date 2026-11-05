import test from 'node:test';import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {prepareClothTopology,createClothState,point3,norm3,sub3} from '../lib/cloth/topology';
import {resolveClothContacts,closestTrianglePoint} from '../lib/cloth/contact';
function fixture(extraLayer=false){
 const positions:number[]=[],triangles:number[]=[],side=10,spacing=.00144;for(let z=0;z<side;z++)for(let x=0;x<side;x++)positions.push(x*spacing,0,z*spacing);
 for(let z=0;z<side-1;z++)for(let x=0;x<side-1;x++){const a=z*side+x;triangles.push(a,a+side,a+1,a+1,a+side,a+side+1);}
 const fixed=Array(side*side).fill(extraLayer?1:0);if(extraLayer){const n=positions.length/3;positions.push(.003,.02,.003,.006,.02,.003,.003,.02,.006);triangles.push(n,n+2,n+1);fixed.push(0,0,0);}
 const topology=prepareClothTopology({restPositions_m:positions,triangles,rodOffsets:[0],rodIndices:[],fixed},[{...DEFAULT_CLOTH_MATERIAL,thickness_m:.0032}]);return{topology,state:createClothState(topology)};
}
test('thick rest shell excludes its connected material neighborhood without moving or shrinking the surface',()=>{
 const {topology,state}=fixture(),old=state.positions_m.slice(),report=resolveClothContacts(topology,state,old,.001,[],true);
 assert.equal(report.selfContacts,0);assert.equal(report.iterations,1);assert.deepEqual(state.positions_m,old);assert.equal(topology.materials[0]!.thickness_m,.0032);
});
test('rest-neighborhood filtering retains actual CCD against disconnected shell layers',()=>{
 const {topology,state}=fixture(true),old=state.positions_m.slice();for(let i=100;i<103;i++){state.positions_m[3*i+1]=-.01;state.velocities_m_s[3*i+1]=-30;}
 const report=resolveClothContacts(topology,state,old,.001,[],true);assert.ok(report.selfContacts>0);for(let i=100;i<103;i++){const p=point3(state.positions_m,i),gap=Math.min(...topology.triangles.slice(0,-1).map(f=>norm3(sub3(p,closestTrianglePoint(p,...f.vertices.map(v=>point3(state.positions_m,v)) as [[number,number,number],[number,number,number],[number,number,number]]).point))));assert.ok(gap>=.0032-1e-8,`vertex ${i} separation ${gap}`);assert.ok(p[1]>0,'crossing remains on its initial side of the fixed sheet');}
 for(let i=0;i<300;i++)assert.equal(state.positions_m[i],old[i]);
});
